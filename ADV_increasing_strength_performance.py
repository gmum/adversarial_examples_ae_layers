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
import joblib
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score

runs_regex = re.compile(r'(\w*)_(\w*)_deep_(wae)_.*?$')


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
        model.load_state_dict(
            torch.load(pre_trained_net, map_location="cuda:" + str(args.gpu)))
        in_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255),
                                 (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
        ])
    elif net_type == 'resnet':
        model = models.ResNet34(num_c=args.num_classes)
        model.load_state_dict(
            torch.load(pre_trained_net, map_location="cuda:" + str(args.gpu)))
        in_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    model.cuda()
    print('load model: ' + net_type)

    # load dataset
    print('load target data: ', dataset)
    train_loader, test_loader = data_loader.getTargetDataSet(
        dataset, args.batch_size, in_transform, args.dataroot)

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
        ae_models = [
            build_ae_for_size(ae_type, sizes, filters, latent_size, lam, bn)
            for sizes, lam in zip(feature_map_sizes, lamb)
        ]
    else:
        ae_models = [
            build_ae_for_size(ae_type, sizes, filters, latent_size, lamb, bn)
            for sizes in feature_map_sizes
        ]

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

    # read in final unsupervised model
    classifier_model = 'IF'
    run_dir = args.run_dir
    name_prefix = 'latent_' if args.latent else ''
    name_prefix += 'rec_error_' if args.rec_error else ''
    name_prefix += 'both_' if args.both else ''
    model_filename = str(run_dir /
                         f'{name_prefix}final_cv_{classifier_model}.joblib')
    assert Path(model_filename).exists()
    if_classfier = joblib.load(model_filename)

    pgd_iters = [(i + 1) * 10 for i in range(20)]
    results_file = run_dir / 'increasing_strength_pgd.pickle'
    if results_file.exists():
        with open(results_file, 'rb') as results_fd:
            results = pickle.load(results_fd)
    else:
        results = {}
    for iters in pgd_iters:
        # calculate only if result is not present already
        key = f'PGD{iters}'
        if key not in results:
            # read in adversarial AE encoded samples
            dataset_names = [f'clean_{key}', f'adv_{key}', f'noisy_{key}']
            datasets = {}
            for name in dataset_names:
                dataset_path = run_dir / f'ae_encoded_{name}.npy'
                if dataset_path.exists():
                    datasets[name] = np.load(str(dataset_path))
                else:
                    print(f'{dataset_path} is missing!')
            test_size = len(datasets[f'clean_{key}'])
            X_test = np.concatenate([
                datasets[f'clean_{key}'],
                datasets[f'adv_{key}'],
                datasets[f'noisy_{key}'],
            ])
            y_test = np.concatenate([
                np.ones(test_size),
                np.zeros(test_size),
                np.ones(test_size),
            ])
            # classify with the unsupervised classifier
            y_pred = if_classfier.predict(X_test)
            try:
                y_scores = if_classfier.decision_function(X_test)
            except:
                y_scores = if_classfier.predict_proba(X_test)[0]
            # calculate scores
            acc = accuracy_score(y_test, y_pred)
            auroc = roc_auc_score(y_test, y_scores)
            results[f'acc_{key}'] = acc
            results[f'auroc_{key}'] = auroc
            # save results
            with open(results_file, 'wb') as results_fd:
                pickle.dump(results, results_fd)

    # gather data to numpy arrays
    xs = np.array(pgd_iters)
    ys = np.zeros(len(xs))
    ys_acc = np.zeros(len(xs))
    for i, iters in enumerate(pgd_iters):
        key = f'PGD{iters}'
        ys[i] = results[f'auroc_{key}']
        ys_acc[i] = results[f'acc_{key}']

    # generate plots
    plot_filename = run_dir / 'increasing_strength_pgd.png'
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 9))
    color = 'tab:red'
    ax1.set_xlabel('PGD iterations')
    ax1.set_ylabel('AUROC', color=color)
    ax1.plot(xs, ys, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy',
                   color=color)  # we already handled the x-label with ax1
    ax2.plot(xs, ys_acc, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    print(f'Writing {plot_filename}')
    fig.savefig(plot_filename, bbox_inches='tight')

    plot_filename = run_dir / 'auroc_increasing_strength_pgd.png'
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 9))
    ax1.set_xlabel('PGD iterations')
    ax1.set_ylabel('AUROC')
    x_ticks = xs[::2]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([str(x) for x in x_ticks])
    ax1.plot(xs, ys, marker='o', linestyle='solid', markeredgecolor='b')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    print(f'Writing {plot_filename}')
    fig.savefig(plot_filename, bbox_inches='tight')


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
    parser.add_argument('--batch_size',
                        type=int,
                        default=100,
                        metavar='N',
                        help='batch size for data loader')
    parser.add_argument('--dataroot',
                        default=str(Path.home() / '.datasets'),
                        help='path to dataset')
    parser.add_argument('--latent', action='store_true', help='train model on the whole latent representation')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--rec_error', action='store_true', help='train model on the reconstruction error AE instead')
    group.add_argument('--both', action='store_true', help='train model on both AEs')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    args = parser.parse_args()
    print(args)
    main()
