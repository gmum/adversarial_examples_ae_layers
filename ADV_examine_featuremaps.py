import re
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
# noinspection PyUnresolvedReferences
from matplotlib.pyplot import close
from scipy.stats import describe
from itertools import cycle

colors = ['green', 'lime', 'seagreen', 'forestgreen',
          'red', 'magenta', 'cyan', 'blue',
          'gold', 'yellow', 'khaki', 'palegoldenrod',
          'black']

# runs_regex = re.compile(r'.*_epochs_\d+$')
# runs_regex = re.compile(r'.*_(wae|ae|vae|waegan)_.*_epochs_\d\d\d$')
runs_regex = re.compile(r'.*_(wae)_.*_epochs_\d\d\d_0$')

adv_types = ['FGSM', 'BIM', 'DeepFool', 'CWL2', 'PGD100']


def main():
    run_dirs = []
    for p in Path('.').iterdir():
        if runs_regex.match(p.name) and p.is_dir():
            run_dirs.append(p)

    for run_dir in run_dirs:
        print(f'Processing {str(run_dir)}')
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

        # print t-SNE visualizations
        # fig, ax = plt.subplots(figsize=(16, 9))
        # fig3d = plt.figure(figsize=(16, 9))
        # ax3d = fig3d.add_subplot(1, 1, 1, projection='3d')
        # for dataset_name, dataset, color in zip(datasets.items(), colors):
        #     # TODO randomize?
        #     # processed_dataset = StandardScaler().fit_transform(dataset)
        #     # processed_dataset = MinMaxScaler().fit_transform(dataset)
        #     processed_dataset = dataset
        #     # ParallelTSNE package seems to be the fastest implementation with 3D support
        #     X_tsned = TSNE(3, n_jobs=8).fit_transform(processed_dataset)
        #     X_tsned_3d = TSNE(n_components=3, n_jobs=8).fit_transform(processed_dataset)
        #     xs = X_tsned[:SAMPLES_PER_DATASET, 0]
        #     ys = X_tsned[:SAMPLES_PER_DATASET, 1]
        #     ax.scatter(xs, ys, c=color, label=dataset_name)
        #     # ax.grid(True)
        #     ax.legend()
        #     xs = X_tsned_3d[:SAMPLES_PER_DATASET, 0]
        #     ys = X_tsned_3d[:SAMPLES_PER_DATASET, 1]
        #     zs = X_tsned_3d[:SAMPLES_PER_DATASET, 2]
        #     ax3d.scatter(xs, ys, zs, c=color, label=dataset_name)
        #     # ax3d.grid(True)
        #     ax3d.legend()
        # plot_filename = f'{str(run_dir)}/tsne_2d.png'
        # plot_3d_filename = f'{str(run_dir)}/tsne_3d.png'
        # print(f'Writing plots to {plot_filename} and {plot_3d_filename}')
        # fig.savefig(plot_filename, bbox_inches='tight')
        # fig3d.savefig(plot_3d_filename, bbox_inches='tight')
        # close(fig)
        # close(fig3d)

        # print statistics for the generated datasets, plot scatter plots
        stats_filename = f'{str(run_dir)}/stats.txt'
        print(f'Calculating and writing stats to {stats_filename}')
        n_features = list(datasets.values())[0].shape[1]
        with open(stats_filename, 'w') as stats_file:
            for dataset_name, dataset in datasets.items():
                no_o, minmax, mean, variance, skewness, kurtosis = describe(dataset, axis=0)
                print('=====================================================', file=stats_file)
                print(f'DATASET: {dataset_name}\nNumber of samples: {no_o}', file=stats_file)
                for i in range(n_features):
                    print(
                        f'[feature {i}] min: {minmax[0][i]} max: {minmax[1][i]} mean: {mean[i]} std: {variance[i] ** 0.5}',
                        file=stats_file)
            print('=====================================================', file=stats_file)

        labels = ['1', '34', '67', '99'] if 'densenet' in str(run_dir) else ['1', '7', '15', '27', '33']

        configs = [('adv', [f'clean_{adv_type}' for adv_type in adv_types] +
                    [f'adv_{adv_type}' for adv_type in adv_types], 500),
                   ('noisy', [f'clean_{adv_type}' for adv_type in adv_types] +
                    [f'noisy_{adv_type}' for adv_type in adv_types], 500),
                   ('all', [f'clean_{adv_type}' for adv_type in adv_types] +
                    [f'adv_{adv_type}' for adv_type in adv_types] +
                    [f'noisy_{adv_type}' for adv_type in adv_types] + ['train'], 200),
                   ]
        for config in configs:
            name, datasets_keys, num_samples = config
            for i in range(int(n_features / 2)):
                f1_i = i * 2
                f2_i = f1_i + 1
                fig, ax = plt.subplots(figsize=(16 * len(adv_types), 9 * len(labels)))
                c_iter = cycle(colors)
                for dataset_name in datasets_keys:
                    processed_dataset = datasets[dataset_name]
                    adv_xs = processed_dataset[:num_samples, f1_i]
                    adv_ys = processed_dataset[:num_samples, f2_i]
                    color = next(c_iter)
                    ax.scatter(adv_xs, adv_ys, label=dataset_name, color=color)
                    ax.grid(True)
                    ax.legend()
                    ax.set_xlabel(f'Layer {labels[i]} reconstruction loss')
                    ax.set_ylabel(f'Layer {labels[i]} latent norm')
                plot_filename = f'{str(run_dir)}/{name}_scatter_{i}.png'
                print(f'Writing {plot_filename}')
                fig.savefig(plot_filename, bbox_inches='tight')
                close(fig)

        num_samples = 500
        for smp_type in ['adv', 'noisy']:
            plot_filename = f'{str(run_dir)}/{smp_type}_scatter.png'
            fig, axis = plt.subplots(int(n_features / 2), len(adv_types), figsize=(16 * len(adv_types), 9 * len(labels)))
            kde_plot_filename = f'{str(run_dir)}/{smp_type}_kde.png'
            kde_fig, kde_axis = plt.subplots(int(n_features / 2), len(adv_types), figsize=(16 * len(adv_types), 9 * len(labels)))
            for i in range(int(n_features / 2)):
                f1_i = i * 2
                f2_i = f1_i + 1
                for j, adv_type in enumerate(adv_types):
                    adv_dataset = datasets[f'{smp_type}_{adv_type}']
                    adv_xs = adv_dataset[:num_samples, f1_i]
                    adv_ys = adv_dataset[:num_samples, f2_i]
                    clean_dataset = datasets[f'clean_{adv_type}']
                    clean_xs = clean_dataset[:num_samples, f1_i]
                    clean_ys = clean_dataset[:num_samples, f2_i]
                    axis[i, j].scatter(adv_xs, adv_ys, color='red' if smp_type == 'adv' else 'gold')
                    axis[i, j].scatter(clean_xs, clean_ys, color='green', alpha=0.75)
                    axis[i, j].set_title(f'{adv_type} on layer {labels[i]}')
                    axis[i, j].locator_params(nbins=5)
                    # kde
                    # kde_axis[i, j].scatter(clean_xs, clean_ys, color='white', alpha=0.2)
                    # kde_axis[i, j].scatter(adv_xs, adv_ys, color='red' if smp_type == 'adv' else 'gold', alpha=0.2)
                    sns.kdeplot(adv_xs, adv_ys, cmap='Reds' if smp_type == 'adv' else 'Oranges', shade=True,
                                shade_lowest=False, ax=kde_axis[i, j])
                    sns.kdeplot(clean_xs, clean_ys, cmap='Greens', shade=True, shade_lowest=False, ax=kde_axis[i, j],
                                alpha=0.55)
                    # sns.kdeplot(adv_xs, adv_ys, cmap='Reds' if smp_type == 'adv' else 'Oranges', ax=kde_axis[i, j])
                    # sns.kdeplot(clean_xs, clean_ys, cmap='Greens', ax=kde_axis[i, j])
                    kde_axis[i, j].set_title(f'{adv_type} on layer {labels[i]}')
                    kde_axis[i, j].locator_params(nbins=5)
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
    main()
