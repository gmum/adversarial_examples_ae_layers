import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import seaborn
from matplotlib import pyplot as plt
from matplotlib.pyplot import close

models = ['densenet', 'resnet']
datasets = ['cifar10', 'cifar100', 'svhn']
adv_types = ['FGSM', 'BIM', 'DeepFool', 'CWL2']

HSPACE = 0.4
WSPACE = 0.5

# FIGURE_ROWS = 2
# FIGURE_SIZE = (16, 9)

FIGURE_ROWS = 1
FIGURE_SIZE = (16, 6)


def main():
    latent_prefix = 'latent_' if args.latent else ''
    # read in RF results for each
    results = {}
    for model in models:
        for dataset in datasets:
            num_layers = 4 if model == 'densenet' else 5
            features_per_layer = None
            for file in args.runs_dir.iterdir():
                if re.match(f'{dataset}_{model}_deep_{args.ae_type}_(.*)', file.name):
                    dir_name = file.name
                    break
            else:
                raise FileNotFoundError(f'Missing results file for: {model}/{dataset}/{args.ae_type}')
            # layer importances plot
            plot_filename = f'{latent_prefix}layer_importances.png'
            plot_file = args.runs_dir / dir_name / plot_filename
            fig, ax = plt.subplots(FIGURE_ROWS, len(adv_types) // FIGURE_ROWS, figsize=FIGURE_SIZE)
            labels = ['1', '66', '132', '197'] if model == 'densenet' else ['1', '7', '15', '27', '33']
            xs = np.arange(num_layers)
            color_iter = iter(seaborn.color_palette())
            for k, adv_type in enumerate(adv_types):
                file_name = f'{latent_prefix}final_cv_RF_known_{adv_type}.joblib'
                file = args.runs_dir / dir_name / file_name
                gs = joblib.load(file)
                rf = gs.best_estimator_['clf']
                importances = rf.feature_importances_
                if features_per_layer is None:
                    assert importances.shape[0] % num_layers == 0
                    features_per_layer = importances.shape[0] // num_layers
                layer_importances = np.zeros(num_layers)
                for i in range(num_layers):
                    layer_importances[i] = importances[i * features_per_layer: (i + 1) * features_per_layer].sum()
                i, j = divmod(k, FIGURE_ROWS)
                chosen_ax = ax[i][j] if FIGURE_ROWS > 1 else ax[i]
                chosen_ax.bar(xs, layer_importances, color=next(color_iter), width=0.4)
                chosen_ax.grid(True)
                chosen_ax.set_xlabel(f'Layer')
                chosen_ax.set_ylabel(f'Importance')
                chosen_ax.set_xticks(xs)
                chosen_ax.set_xticklabels(labels)
                chosen_ax.set_title(f'{adv_type}')
            plt.subplots_adjust(wspace=WSPACE, hspace=HSPACE)
            fig.savefig(plot_file, bbox_inches='tight')
            close(fig)
            # feature importances plot
            plot_filename = f'{latent_prefix}feature_importances.png'
            plot_file = args.runs_dir / dir_name / plot_filename
            fig, ax = plt.subplots(FIGURE_ROWS, len(adv_types) // FIGURE_ROWS, figsize=FIGURE_SIZE)
            xs = np.arange(features_per_layer)
            labels = [str(i) for i in xs] if len(xs) > 2 else ['Rec. error', 'L2 norm']
            color_iter = iter(seaborn.color_palette())
            for k, adv_type in enumerate(adv_types):
                file_name = f'{latent_prefix}final_cv_RF_known_{adv_type}.joblib'
                file = args.runs_dir / dir_name / file_name
                gs = joblib.load(file)
                rf = gs.best_estimator_['clf']
                importances = rf.feature_importances_
                feature_importances = np.zeros(features_per_layer)
                for i in range(features_per_layer):
                    feature_mask = np.zeros(importances.shape[0])
                    for j in range(num_layers):
                        feature_mask[j * features_per_layer + i] = 1.0
                    feature_importances[i] = (importances * feature_mask).sum()
                i, j = divmod(k, FIGURE_ROWS)
                chosen_ax = ax[i][j] if FIGURE_ROWS > 1 else ax[i]
                chosen_ax.bar(xs, feature_importances, color=next(color_iter), width=0.2)
                chosen_ax.grid(True)
                chosen_ax.set_xticks(xs)
                chosen_ax.set_xticklabels(labels)
                # chosen_ax.set_xlabel(f'Feature num')
                chosen_ax.set_ylabel(f'Importance')
                chosen_ax.set_title(f'{adv_type}')
            plt.subplots_adjust(wspace=WSPACE, hspace=HSPACE)
            fig.savefig(plot_file, bbox_inches='tight')
            close(fig)


if __name__ == '__main__':
    seaborn.set()
    SMALL_SIZE = 14
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 22
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    parser = argparse.ArgumentParser()
    parser.add_argument('runs_dir', type=Path)
    parser.add_argument('--ae_type', default='vae')
    parser.add_argument('--latent', action='store_true', help='Train model on the whole latent representation')
    args = parser.parse_args()
    main()
