import argparse
import re
from pathlib import Path

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
# noinspection PyUnresolvedReferences
from matplotlib.pyplot import close
from torchvision import transforms

import data_loader
import models
from ADV_train_featuremaps_AE import build_ae_for_size
from lib import adversary
from models.vae import ConvVAE
from models.wae import ConvWAE
from adv_attacks import pgd_linf

adv_types = ['FGSM', 'BIM', 'DeepFool', 'PGD100']

# runs_regex = re.compile(r'(\w*)_(\w*)_deep_(wae|ae|vae|waegan)_.*?_\d\d\d$')
runs_regex = re.compile(r'(\w*)_(\w*)_deep_(wae)_.*?_\d\d\d_\d$')


def main():
    run_dir = args.run_dir
    run_name = run_dir.name
    matches = runs_regex.match(run_name)
    dataset = matches.group(1)
    net_type = matches.group(2)
    ae_type = matches.group(3)
    args.num_classes = 100 if dataset == 'cifar100' else 10

    # set the path to pre-trained model and output
    pre_trained_net = './pre_trained/' + net_type + '_' + dataset + '.pth'
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)
    device = f'cuda:{args.gpu}'

    # load networks
    if net_type == 'densenet':
        model = models.DenseNet3(100, int(args.num_classes))
        model.load_state_dict(torch.load(pre_trained_net, map_location="cuda:" + str(args.gpu)))
        in_transform = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255),
                                                                (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)), ])
    elif net_type == 'resnet':
        model = models.ResNet34(num_c=args.num_classes)
        model.load_state_dict(torch.load(pre_trained_net, map_location="cuda:" + str(args.gpu)))
        in_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
             ])
    model.cuda()
    print('load model: ' + net_type)

    # load dataset
    print('load target data: ', dataset)
    train_loader, test_loader = data_loader.getTargetDataSet(dataset, args.batch_size, in_transform, args.dataroot)

    # get the feature maps' sizes
    model.eval()
    temp_x = torch.rand(1, 3, 32, 32).cuda()
    feature_maps = model.feature_list(temp_x)[1]
    feature_map_sizes = []
    for out in feature_maps:
        feature_map_sizes.append(out.size()[1:])

    criterion = torch.nn.CrossEntropyLoss()
    ae_criterion = torch.nn.MSELoss(reduction='none')
    ae_type = ae_type
    filters = [128, 128, 128]
    latent_size = 64
    bn = False
    if ae_type == 'vae':
        lamb = 1e-7
    elif ae_type == 'wae':
        lamb = 1e-4
    elif ae_type == 'waegan':
        if net_type == 'densenet':
            lamb = [5e-5, 1e-5, 1e-5, 1e-3]
        elif net_type == 'resnet':
            lamb = [5e-5, 5e-6, 5e-6, 2e-5, 5e-4]
    else:
        lamb = None

    if ae_type == 'waegan':
        ae_models = [build_ae_for_size(ae_type, sizes, filters, latent_size, lam, bn) for sizes, lam in
                     zip(feature_map_sizes, lamb)]
    else:
        ae_models = [build_ae_for_size(ae_type, sizes, filters, latent_size, lamb, bn) for sizes in feature_map_sizes]

    # read in model and autoencoder models
    models_filename = f'{run_name}/models.pth'
    state_dict = torch.load(models_filename)
    for i, _ in enumerate(ae_models):
        if isinstance(ae_models[i], torch.nn.Module):
            ae_models[i].load_state_dict(state_dict[f'model_{i}'])
        else:
            for j in range(len(ae_models[i])):
                ae_models[i][j].load_state_dict(state_dict[f'model_{i}'][j])
    for ae_model in ae_models:
        if isinstance(ae_model, torch.nn.Module):
            ae_model.eval()
        else:
            for m in ae_model:
                m.eval()

    # read in autoencoded datasets for visualization
    dataset_names = ['train']
    for adv_type in adv_types:
        dataset_names.append(f'clean_{adv_type}')
        dataset_names.append(f'adv_{adv_type}')
        dataset_names.append(f'noisy_{adv_type}')
    datasets = {}
    for name in dataset_names:
        dataset_path = run_dir / f'ae_encoded_{name}.npy'
        if dataset_path.exists():
            datasets[name] = np.load(str(dataset_path))
        else:
            print(f'{dataset_path} is missing!')

    samples_iterate = 100
    test_loader_iter = iter(test_loader)
    X, y = next(test_loader_iter)
    X, y = X.cuda(), y.cuda()
    # plot
    plot_filename = f'{str(run_dir)}/adv_visualize.png'
    kde_plot_filename = f'{str(run_dir)}/adv_visualize_kde.png'
    labels = ['1', '34', '67', '99'] if 'densenet' in str(run_dir) else ['1', '7', '15', '27', '33']
    fig, ax = plt.subplots(len(ae_models), len(adv_types), figsize=(16 * len(adv_types), 9 * len(labels)))
    kde_fig, kde_ax = plt.subplots(len(ae_models), len(adv_types), figsize=(16 * len(adv_types), 9 * len(labels)))
    clean_samples = 250
    for l, adv_type in enumerate(adv_types):
        print(f'generating adv type: {adv_type}')
        k = l
        while True:
            sample_x, sample_y = X[k].unsqueeze(0), y[k].unsqueeze(0)
            # run the sample through the model
            y_pred, feature_list = model.feature_list(sample_x)
            # check if it is classified properly, get another sample if not
            if torch.argmax(y_pred) == sample_y:
                inputs = sample_x.repeat(samples_iterate, 1, 1, 1)
            else:
                k += 1
                print(f'torch.argmax(y_pred): {torch.argmax(y_pred)}; '
                      f'sample_y: {sample_y}; increasing k to: {k}')
                continue
            # generate adversarial examples
            inputs.requires_grad_()
            output = model(inputs)
            loss = criterion(output, y)
            model.zero_grad()
            loss.backward()
            if net_type == 'densenet':
                min_pixel = -1.98888885975
                max_pixel = 2.12560367584
            elif net_type == 'resnet':
                min_pixel = -2.42906570435
                max_pixel = 2.75373125076
            if adv_type == 'FGSM':
                # increasing epsilon
                # adv_noise = 0.05
                adv_noise = torch.linspace(0.0, 0.1, steps=samples_iterate, device=device).view(-1, 1, 1, 1)
                gradient = torch.ge(inputs.grad.data, 0)
                gradient = (gradient.float() - 0.5) * 2
                if net_type == 'densenet':
                    gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0 / 255.0))
                    gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1 / 255.0))
                    gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7 / 255.0))
                else:
                    gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
                    gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
                    gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                         gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
                # adv_data = torch.add(inputs.data, adv_noise, gradient)
                perturbations = gradient * adv_noise
                adv_data = inputs.data + perturbations
                adv_data = torch.clamp(adv_data, min_pixel, max_pixel).detach()
            elif adv_type == 'BIM':
                # adv_noise = 0.01
                adv_noise = torch.linspace(0.0, 0.02, steps=samples_iterate, device=device).view(-1, 1, 1, 1)
                gradient = torch.sign(inputs.grad.data)
                for _ in range(5):
                    perturbations = gradient * adv_noise
                    inputs = inputs + perturbations
                    inputs = torch.clamp(inputs, min_pixel, max_pixel).detach()
                    inputs.requires_grad_()
                    output = model(inputs)
                    loss = criterion(output, y)
                    model.zero_grad()
                    loss.backward()
                    gradient = torch.sign(inputs.grad.data)
                    if net_type == 'densenet':
                        gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                             gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0 / 255.0))
                        gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                             gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1 / 255.0))
                        gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                             gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7 / 255.0))
                    else:
                        gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
                                             gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
                        gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
                                             gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
                        gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
                                             gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
                    perturbations = gradient * adv_noise
                    adv_data = inputs.data + perturbations
                    adv_data = torch.clamp(adv_data, min_pixel, max_pixel).detach()
            elif adv_type == 'DeepFool':
                if net_type == 'resnet':
                    if dataset == 'cifar10':
                        adv_noise_base = 0.18
                    elif dataset == 'cifar100':
                        adv_noise_base = 0.03
                    else:
                        adv_noise_base = 0.1
                else:
                    if dataset == 'cifar10':
                        adv_noise_base = 0.6
                    elif dataset == 'cifar100':
                        adv_noise_base = 0.1
                    else:
                        adv_noise_base = 0.5
                adv_data = inputs
                for i, adv_noise in enumerate(np.linspace(0.0, adv_noise_base * 2, samples_iterate)):
                    sample_x_expanded = sample_x.expand(2, -1, -1, -1)
                    sample_y_expanded = sample_y.expand(2)
                    _, adv_example = adversary.deepfool(model, sample_x_expanded.data.clone(),
                                                        sample_y_expanded.data.cpu(), \
                                                        args.num_classes, step_size=adv_noise, train_mode=False)
                    adv_example = adv_example.cuda()
                    adv_data[i] = adv_example[0]
            elif adv_type.startswith('PGD'):
                pgd_iters = int(re.match('PGD(\d+)', adv_type).group(1))
                adv_noise_base = 0.05
                adv_data = inputs
                for i, adv_noise in enumerate(np.linspace(0.0, adv_noise_base * 2, samples_iterate)):
                    sample_x_expanded = sample_x.expand(2, -1, -1, -1)
                    sample_y_expanded = sample_y.expand(2)
                    perturbation = pgd_linf(model, sample_x_expanded, sample_y_expanded, adv_noise, 1e-2, 100)
                    adv_data[i] = sample_x_expanded[0] + perturbation[0]
            else:
                raise ValueError(f'Unsupported adv_type value: {adv_type}')
            y_adv, adv_feature_list = model.feature_list(adv_data)
            if torch.argmax(y_adv[-1]) == sample_y:
                k += 1
                print(f'torch.argmax(y_adv[-1]): {torch.argmax(y_adv[-1])}; '
                      f'sample_y: {sample_y}; increasing k to: {k}')
                continue
            break
        # get AE features for the created adv examples
        features = []
        for j, (ae_model, feature_map) in enumerate(zip(ae_models, adv_feature_list)):
            feature_map = feature_map.detach()
            if isinstance(ae_model, ConvVAE):
                fm_recon, mu, logvar, z = ae_model.forward(feature_map)
                loss, rec_loss, div_loss = ae_model.calculate_loss(feature_map, fm_recon, ae_criterion, mu,
                                                                   logvar)
                mu_norm = torch.norm(mu, dim=1)
                logvar_norm = torch.norm(logvar, dim=1)
                features.append(
                    torch.cat([rec_loss.mean(dim=(1, 2, 3)).unsqueeze(1), mu_norm.unsqueeze(1),
                               logvar_norm.unsqueeze(1)], dim=1).detach().cpu().numpy())
            elif isinstance(ae_model, ConvWAE):
                fm_recon, z = ae_model.forward(feature_map)
                z_sigma = ae_model.z_var ** 0.5
                z_gen = torch.empty((feature_map.size(0), ae_model.z_dim)).normal_(std=z_sigma).cuda()
                loss, rec_loss, dis_loss = ae_model.calculate_loss(feature_map, fm_recon, ae_criterion,
                                                                   z_gen, z)
                latent_norm = torch.norm(z, dim=1)
                features.append(
                    torch.cat([rec_loss.mean(dim=(1, 2, 3)).unsqueeze(1), latent_norm.unsqueeze(1)],
                              dim=1).detach().cpu().numpy())
            elif isinstance(ae_model, tuple):
                z = ae_model[0](feature_map)
                fm_recon = ae_model[1](z)
                dis_output = ae_model[2](z)
                latent_norm = torch.norm(z, dim=1)
                rec_loss = ae_criterion(fm_recon, feature_map).mean(dim=(1, 2, 3))
                features.append(
                    torch.cat(
                        [rec_loss.unsqueeze(1), latent_norm.unsqueeze(1), dis_output],
                        dim=1).detach().cpu().numpy())
            else:
                fm_recon, z = ae_model.forward(feature_map)
                loss, rec_loss = ae_model.calculate_loss(feature_map, fm_recon, ae_criterion)
                latent_norm = torch.norm(z, dim=1)
                features.append(
                    torch.cat([rec_loss.mean(dim=(1, 2, 3)).unsqueeze(1), latent_norm.unsqueeze(1)],
                              dim=1).detach().cpu().numpy())
        # generate and save figures
        colors = np.zeros((adv_data.shape[0], 3))
        colors[:, 1] = np.linspace(1, 0, adv_data.shape[0])
        colors[:, 0] = 1
        for j in range(len(features)):
            # additionally plot multiple points (or KDE) from training/clean data
            f1_i = j * 2
            f2_i = f1_i + 1
            clean_xs = datasets['train'][:, f1_i][:clean_samples]
            clean_ys = datasets['train'][:, f2_i][:clean_samples]
            ax[j, l].scatter(clean_xs,
                             clean_ys,
                             color='green')
            sns.kdeplot(clean_xs, clean_ys, cmap='Greens', shade=True, shade_lowest=False, ax=kde_ax[j, l], alpha=1.0)
            ax[j, l].scatter(features[j][:, 0], features[j][:, 1], color=colors)
            kde_ax[j, l].scatter(features[j][:, 0], features[j][:, 1], color=colors)
            ax[j, l].grid(True)
            kde_ax[j, l].grid(True)
            ax[j, l].set_title(f'{adv_type} on layer {labels[j]}')
            kde_ax[j, l].set_title(f'{adv_type} on layer {labels[j]}')
            ax[j, l].locator_params(nbins=5)
            kde_ax[j, l].locator_params(nbins=5)

    print(f'Writing {plot_filename}')
    fig.savefig(plot_filename, bbox_inches='tight')
    close(fig)
    print(f'Writing {kde_plot_filename}')
    kde_fig.savefig(kde_plot_filename, bbox_inches='tight')
    close(kde_fig)

    print(f'FINISHED {str(run_dir)}!')


if __name__ == '__main__':
    sns.set()
    SMALL_SIZE = 28
    MEDIUM_SIZE = 30
    BIGGER_SIZE = 42
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--batch_size', type=int, default=100, metavar='N', help='batch size for data loader')
    parser.add_argument('--dataroot', default=str(Path.home() / '.datasets'), help='path to dataset')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    args = parser.parse_args()
    print(args)
    main()
