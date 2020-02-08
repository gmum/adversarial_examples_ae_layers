import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

import data_loader
import models
from models.ae import ConvAE
from models.vae import ConvVAE
from models.wae import ConvWAE
from models.waegan import ConvWAEGAN_Decoder, ConvWAEGAN_Encoder, ConvWAEGAN_Discriminator


def custom_schedule(optimizer, epoch):
    change = False
    if epoch == 75:
        change = 0.5
    elif epoch == 100:
        change = 0.2
    elif epoch == 125:
        change = 0.1
    if change:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * change


def build_ae_for_size(ae_class, sizes, filters, latent_size, lamb=None, bn=False, info=False):
    l = math.floor(math.log(min(sizes[1:]), 2))
    l = min(l, 3)
    size_preserving_l = len(filters) - l
    kernels = [3] * size_preserving_l + [4] * l
    strides = [1] * size_preserving_l + [2] * l
    paddings = [1] * size_preserving_l + [1] * l
    if ae_class == 'vae':
        ae = ConvVAE(sizes, (filters, list(reversed(filters[:-1]))), (kernels, kernels),
                     strides=(strides, strides), paddings=(paddings, paddings), latent_size=latent_size,
                     lamb=lamb, bn=bn).cuda()
    elif ae_class == 'ae':
        assert lamb is None
        ae = ConvAE(sizes, (filters, list(reversed(filters[:-1]))), (kernels, kernels),
                    strides=(strides, strides), paddings=(paddings, paddings), latent_size=latent_size, bn=bn).cuda()
    elif ae_class == 'wae':
        ae = ConvWAE(sizes, (filters, list(reversed(filters[:-1]))), (kernels, kernels),
                     strides=(strides, strides), paddings=(paddings, paddings), latent_size=latent_size, bn=bn,
                     lamb=lamb, z_var=2.0).cuda()
    elif ae_class == 'waegan':
        encoder = ConvWAEGAN_Encoder(sizes, (filters, list(reversed(filters[:-1]))), (kernels, kernels),
                                     strides=(strides, strides), paddings=(paddings, paddings), latent_size=latent_size,
                                     lamb=lamb, bn=bn).cuda()
        decoder = ConvWAEGAN_Decoder(sizes, (filters, list(reversed(filters[:-1]))), (kernels, kernels),
                                     strides=(strides, strides), paddings=(paddings, paddings), latent_size=latent_size,
                                     lamb=lamb, bn=bn).cuda()
        discriminator = ConvWAEGAN_Discriminator(latent_size=latent_size, lamb=lamb).cuda()
        ae = (encoder, decoder, discriminator)
    else:
        raise ValueError('wrong ae_class parameter value')
    if info:
        print(f'------------------------------')
        print(f'Built {ae_class} with parameters:')
        print(f'input/output size: {tuple(sizes)}')
        print(f'layers: {len(filters)} - {filters}')
        if ae_class == 'waegan':
            print(f'pre-latent: {ae[0].pre_latent_size}')
        else:
            print(f'pre-latent: {ae.pre_latent_size}')
        print(f'latent size: {latent_size}')
    return ae


def main():
    # set the path to pre-trained model and output
    pre_trained_net = './pre_trained/' + args.net_type + '_' + args.dataset + '.pth'
    args.outf = args.outf + args.net_type + '_' + args.dataset + '/'
    if os.path.isdir(args.outf) == False:
        os.mkdir(args.outf)
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)
    # check the in-distribution dataset
    if args.dataset == 'cifar100':
        args.num_classes = 100

    # load networks
    if args.net_type == 'densenet':
        model = models.DenseNet3(100, int(args.num_classes))
        model.load_state_dict(torch.load(pre_trained_net, map_location="cuda:" + str(args.gpu)))
        in_transform = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255),
                                                                (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)), ])
    elif args.net_type == 'resnet':
        model = models.ResNet34(num_c=args.num_classes)
        model.load_state_dict(torch.load(pre_trained_net, map_location="cuda:" + str(args.gpu)))
        in_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
             ])
    model.cuda()
    print('load model: ' + args.net_type)

    # load dataset
    print('load target data: ', args.dataset)
    train_loader, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, in_transform, args.dataroot)
    test_clean_data, test_adv_data, test_noisy_data, test_label = {}, {}, {}, {}
    clean_loaders, adv_loaders, noisy_loaders = {}, {}, {}
    adv_types = ['FGSM', 'BIM', 'DeepFool', 'CWL2']
    for adv_type in adv_types:
        test_clean_data[adv_type] = torch.load(
            args.outf + 'clean_data_%s_%s_%s.pth' % (args.net_type, args.dataset, adv_type))
        test_adv_data[adv_type] = torch.load(
            args.outf + 'adv_data_%s_%s_%s.pth' % (args.net_type, args.dataset, adv_type))
        test_noisy_data[adv_type] = torch.load(
            args.outf + 'noisy_data_%s_%s_%s.pth' % (args.net_type, args.dataset, adv_type))
        test_label[adv_type] = torch.load(args.outf + 'label_%s_%s_%s.pth' % (args.net_type, args.dataset, adv_type))
        clean_loaders[adv_type] = torch.utils.data.DataLoader(test_clean_data[adv_type], batch_size=args.batch_size)
        adv_loaders[adv_type] = torch.utils.data.DataLoader(test_adv_data[adv_type], batch_size=args.batch_size)
        noisy_loaders[adv_type] = torch.utils.data.DataLoader(test_noisy_data[adv_type], batch_size=args.batch_size)

    # get the feature maps' sizes
    model.eval()
    temp_x = torch.rand(1, 3, 32, 32).cuda()
    feature_maps = model.feature_list(temp_x)[1]
    feature_map_sizes = []
    for out in feature_maps:
        feature_map_sizes.append(out.size()[1:])

    opt_class = torch.optim.Adam
    criterion = torch.nn.MSELoss(reduction='none')
    ae_type = args.ae_type
    filters = [128, 128, 128]
    arch_text = str(filters).replace(', ', '_').replace('(', '').replace(')', '')
    latent_size = 64
    bn = False
    if ae_type == 'vae':
        lamb = 1e-7
    elif ae_type == 'wae':
        lamb = 1e-4
    elif ae_type == 'waegan':
        if args.net_type == 'densenet':
            lamb = [5e-5, 1e-5, 1e-5, 1e-3]
        elif args.net_type == 'resnet':
            lamb = [5e-5, 5e-6, 5e-6, 2e-5, 5e-4]
    else:
        lamb = None
    lr = 1e-3

    for run_n in range(args.runs):

        if ae_type == 'waegan':
            ae_models = [build_ae_for_size(ae_type, sizes, filters, latent_size, lam, bn, info=True) for sizes, lam in
                         zip(feature_map_sizes, lamb)]
            ae_opts = []
            for ae_model in ae_models:
                ae_opts.append(tuple([opt_class(ae_model[i].parameters(), lr) for i in range(len(ae_model))]))
        else:
            ae_models = [build_ae_for_size(ae_type, sizes, filters, latent_size, lamb, bn, info=True) for sizes in
                         feature_map_sizes]
            ae_opts = [opt_class(ae_model.parameters(), lr) for ae_model in ae_models]

        run_name = (f'{args.dataset}_{args.net_type}_deep_{ae_type}_arch_{arch_text}_bn_{bn}' \
                    f'_latent_{latent_size}_lamb_{lamb}_lr_{lr}_bs_{args.batch_size}_epochs_{args.epochs}_{run_n}') \
            .replace('.', '_').replace(',', '_')
        print(f'Run name: {run_name}')
        writer = SummaryWriter(run_name)
        for ae_model in ae_models:
            if isinstance(ae_model, torch.nn.Module):
                ae_model.train()
            else:
                for m in ae_model:
                    m.train()
        models_filename = f'{run_name}/models.pth'
        models_file = Path(models_filename)
        if not models_file.exists():
            current_x = 0
            with tqdm(initial=current_x, unit_scale=True, dynamic_ncols=True) as pbar:
                for epoch in range(args.epochs):
                    for ae_model, ae_opt in zip(ae_models, ae_opts):
                        if isinstance(ae_model, tuple):
                            encoder_o, decoder_o, discriminator_o = ae_opt
                            custom_schedule(encoder_o, epoch)
                            custom_schedule(decoder_o, epoch)
                            custom_schedule(discriminator_o, epoch)
                        elif isinstance(ae_model, ConvWAE):
                            custom_schedule(ae_opt, epoch)
                        elif isinstance(ae_model, ConvVAE):
                            custom_schedule(ae_opt, epoch)
                    for X, _ in train_loader:
                        X = X.cuda()
                        y_pred, feature_list = model.feature_list(X)
                        for i, (feature_map, ae_model, ae_opt) in enumerate(zip(feature_list, ae_models, ae_opts)):
                            feature_map = feature_map.detach()
                            if isinstance(ae_model, tuple):
                                encoder, decoder, discriminator = ae_model
                                encoder_o, decoder_o, discriminator_o = ae_opt
                                z = encoder(feature_map)
                                fm_recon = decoder(z)
                                # train discriminator
                                ones = torch.ones((feature_map.size(0), 1), device='cuda')
                                zeros = torch.zeros((feature_map.size(0), 1), device='cuda')
                                z_sigma = encoder.z_var ** 0.5
                                z_fake = torch.empty((X.size(0), discriminator.z_dim), device='cuda').normal_(std=z_sigma)
                                dis_fake = discriminator(z_fake)
                                dis_real = discriminator(z)
                                fake_loss = F.binary_cross_entropy_with_logits(dis_fake, ones)
                                real_loss = F.binary_cross_entropy_with_logits(dis_real, zeros)
                                d_loss = discriminator.lamb * (fake_loss + real_loss)
                                discriminator_o.zero_grad()
                                d_loss.backward(retain_graph=True)
                                discriminator_o.step()

                                # train autoencoder
                                # rec_losses = criterion(fm_recon, feature_map).sum(dim=(1, 2, 3))
                                rec_losses = criterion(fm_recon, feature_map).mean(dim=(1, 2, 3))
                                rec_loss = rec_losses.mean()
                                ae_d_loss = F.binary_cross_entropy_with_logits(dis_real, ones)
                                ae_loss = rec_loss + discriminator.lamb * ae_d_loss
                                encoder_o.zero_grad()
                                decoder_o.zero_grad()
                                ae_loss.backward()
                                encoder_o.step()
                                decoder_o.step()

                                writer.add_scalar(f'AE {i}/AE Loss', ae_loss.item(), global_step=current_x)
                                writer.add_scalar(f'AE {i}/AE Reconstruction loss', rec_loss.item(), global_step=current_x)
                                writer.add_scalar(f'AE {i}/AE discriminator loss', ae_d_loss.item(), global_step=current_x)
                                writer.add_scalar(f'AE {i}/Discriminator loss', d_loss.item(),
                                                  global_step=current_x)
                                writer.add_scalar(f'AE {i}/Discriminator fake loss', fake_loss.item(),
                                                  global_step=current_x)
                                writer.add_scalar(f'AE {i}/Discriminator mean real output',
                                                  torch.sigmoid(dis_real).mean().item(), global_step=current_x)
                                writer.add_scalar(f'AE {i}/Discriminator mean fake output',
                                                  torch.sigmoid(dis_fake).mean().item(), global_step=current_x)
                            else:
                                if isinstance(ae_model, ConvVAE):
                                    fm_recon, mu, logvar, z = ae_model.forward(feature_map)
                                    loss, rec_loss, div_loss = ae_model.calculate_loss(feature_map, fm_recon, criterion, mu,
                                                                                       logvar)
                                elif isinstance(ae_model, ConvWAE):
                                    fm_recon, z = ae_model.forward(feature_map)
                                    z_sigma = ae_model.z_var ** 0.5
                                    z_gen = torch.empty((feature_map.size(0), ae_model.z_dim)).normal_(
                                        std=z_sigma).cuda()
                                    loss, rec_loss, dis_loss = ae_model.calculate_loss(feature_map, fm_recon, criterion,
                                                                                       z_gen,
                                                                                       z)
                                else:
                                    fm_recon, z = ae_model.forward(feature_map)
                                    loss, rec_loss = ae_model.calculate_loss(feature_map, fm_recon, criterion)

                                ae_opt.zero_grad()
                                loss.backward()
                                ae_opt.step()

                                if isinstance(ae_model, ConvVAE):
                                    writer.add_scalar(f'AE {i}/Loss', loss.item(), global_step=current_x)
                                    writer.add_scalar(f'AE {i}/Reconstruction loss', rec_loss.mean().item(),
                                                      global_step=current_x)
                                    writer.add_scalar(f'AE {i}/Divergence loss', div_loss.item(), global_step=current_x)
                                elif isinstance(ae_model, ConvWAE):
                                    writer.add_scalar(f'AE {i}/Loss', loss.item(), global_step=current_x)
                                    writer.add_scalar(f'AE {i}/Reconstruction loss', rec_loss.mean().item(),
                                                      global_step=current_x)
                                    writer.add_scalar(f'AE {i}/Discrepancy loss', dis_loss.item(), global_step=current_x)
                                else:
                                    writer.add_scalar(f'AE {i}/Loss', loss.item(), global_step=current_x)
                        current_x += 1
                        pbar.set_description(f'Epoch {epoch}')
                        pbar.update()

            # save models
            state_dict = {}
            for i, ae_model in enumerate(ae_models):
                if isinstance(ae_model, torch.nn.Module):
                    state_dict[f'model_{i}'] = ae_model.state_dict()
                else:
                    state_dict[f'model_{i}'] = tuple([ae_model[i].state_dict() for i in range(len(ae_model))])
            torch.save(state_dict, models_filename)
        else:
            state_dict = torch.load(models_filename)
            for i, _ in enumerate(ae_models):
                if isinstance(ae_models[i], torch.nn.Module):
                    ae_models[i].load_state_dict(state_dict[f'model_{i}'])
                else:
                    for j in range(len(ae_models[i])):
                        ae_models[i][j].load_state_dict(state_dict[f'model_{i}'][j])

        # push entire datasets through the (trained) classifier and trained autoencoder and save the results
        for ae_model in ae_models:
            if isinstance(ae_model, torch.nn.Module):
                ae_model.eval()
            else:
                for m in ae_model:
                    m.eval()

        # order must be preserved - alternatively use tuples and one list
        loaders = [train_loader]
        dataset_names = ['train']
        datasets = []
        for adv_type in adv_types:
            loaders.append(clean_loaders[adv_type])
            dataset_names.append(f'clean_{adv_type}')
            loaders.append(adv_loaders[adv_type])
            dataset_names.append(f'adv_{adv_type}')
            loaders.append(noisy_loaders[adv_type])
            dataset_names.append(f'noisy_{adv_type}')
        with torch.no_grad():
            for name, loader in zip(dataset_names, loaders):
                filename = f'{run_name}/ae_encoded_{name}.npy'
                if not Path(filename).exists():
                    final_features = []
                    with tqdm(initial=0, unit_scale=True, dynamic_ncols=True) as pbar:
                        for t in loader:
                            if isinstance(t, torch.Tensor):
                                X = t
                            else:
                                X = t[0]
                            X = X.cuda()
                            y_pred, feature_list = model.feature_list(X)
                            features = []
                            for i, (feature_map, ae_model) in enumerate(zip(feature_list, ae_models)):
                                feature_map = feature_map.detach()
                                if isinstance(ae_model, ConvVAE):
                                    fm_recon, mu, logvar, z = ae_model.forward(feature_map)
                                    loss, rec_loss, div_loss = ae_model.calculate_loss(feature_map, fm_recon, criterion, mu,
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
                                    loss, rec_loss, dis_loss = ae_model.calculate_loss(feature_map, fm_recon, criterion,
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
                                    rec_loss = criterion(fm_recon, feature_map).mean(dim=(1, 2, 3))
                                    features.append(
                                        torch.cat(
                                            [rec_loss.unsqueeze(1), latent_norm.unsqueeze(1), dis_output],
                                            dim=1).detach().cpu().numpy())
                                else:
                                    fm_recon, z = ae_model.forward(feature_map)
                                    loss, rec_loss = ae_model.calculate_loss(feature_map, fm_recon, criterion)
                                    latent_norm = torch.norm(z, dim=1)
                                    features.append(
                                        torch.cat([rec_loss.mean(dim=(1, 2, 3)).unsqueeze(1), latent_norm.unsqueeze(1)],
                                                  dim=1).detach().cpu().numpy())
                            final_features.append(np.concatenate(features, axis=1))
                            pbar.update()
                    autoencoded_set = np.concatenate(final_features, axis=0)
                    datasets.append(autoencoded_set)
                    np.save(filename, autoencoded_set)
                    print(f'Saving {name} encoded dataset to {filename}')
                else:
                    print(f'Found {filename}')
                    datasets.append(np.load(filename))
                # and save the entire latent encoding
                filename = f'{run_name}/latent_{name}.npy'
                if not Path(filename).exists():
                    final_features = []
                    with tqdm(initial=0, unit_scale=True, dynamic_ncols=True) as pbar:
                        for t in loader:
                            if isinstance(t, torch.Tensor):
                                X = t
                            else:
                                X = t[0]
                            X = X.cuda()
                            y_pred, feature_list = model.feature_list(X)
                            features = []
                            for i, (feature_map, ae_model) in enumerate(zip(feature_list, ae_models)):
                                feature_map = feature_map.detach()
                                if isinstance(ae_model, ConvVAE):
                                    fm_recon, mu, logvar, z = ae_model.forward(feature_map)
                                elif isinstance(ae_model, ConvWAE):
                                    fm_recon, z = ae_model.forward(feature_map)
                                elif isinstance(ae_model, tuple):
                                    z = ae_model[0](feature_map)
                                else:
                                    fm_recon, z = ae_model.forward(feature_map)
                                features.append(z.detach().cpu().numpy())
                            final_features.append(np.concatenate(features, axis=1))
                            pbar.update()
                    autoencoded_set = np.concatenate(final_features, axis=0)
                    datasets.append(autoencoded_set)
                    np.save(filename, autoencoded_set)
                    print(f'Saving {name} latent encoded dataset to {filename}')
                else:
                    print(f'Found {filename}')

        # test the model performance when features are replaced by the autoencoded features
        def test_ae_dataset(loader):
            model.eval()
            for ae_model in ae_models:
                if isinstance(ae_model, torch.nn.Module):
                    ae_model.eval()
                else:
                    for m in ae_model:
                        m.eval()
            with torch.no_grad():
                with tqdm(initial=0, unit_scale=True, dynamic_ncols=True) as pbar:
                    total = 0
                    keys = ['original', 'all'] + [i + 1 for i in range(len(ae_models))]
                    correct = {k: 0 for k in keys}
                    masks = {k: [False] * len(ae_models) for k in keys}
                    masks['all'] = [True] * len(ae_models)
                    for i in range(len(ae_models)):
                        masks[i + 1][i] = True
                    for X, y in loader:
                        X = X.cuda()
                        y = y.cuda()
                        total += y.size(0)
                        for k in keys:
                            y_pred = model.ae_replaced_forward(X, ae_models, masks[k])
                            y_pred_max = y_pred.argmax(dim=1)
                            correct[k] += (y_pred_max == y).sum().item()
                        pbar.update()
            accs = {k: correct[k] / total for k in keys}
            return accs

        train_accs = test_ae_dataset(train_loader)
        print(f'Trainset accuracy:')
        for k, v in train_accs.items():
            print(f'{k}: {v}')
        test_accs = test_ae_dataset(test_loader)
        print(f'Testset accuracy:')
        for k, v in test_accs.items():
            print(f'{k}: {v}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100, metavar='N', help='batch size for data loader')
    parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn')
    parser.add_argument('--dataroot', default=str(Path.home() / '.datasets'), help='path to dataset')
    parser.add_argument('--outf', default='./adv_output/', help='folder to output results')
    parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
    parser.add_argument('--net_type', required=True, help='resnet | densenet')
    parser.add_argument('--ae_type', required=True, help='ae | vae | wae | waegan')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--epochs', default=150, help='epochs of AE training')
    parser.add_argument('--runs', default=5, help='number of runs')
    args = parser.parse_args()
    print(args)
    main()
