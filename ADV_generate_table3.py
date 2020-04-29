import argparse
import re
from pathlib import Path

import sys
from tabulate import tabulate

parser = argparse.ArgumentParser()
parser.add_argument('runs_dir', type=Path)
parser.add_argument('--classifier_type', default='SVC')
parser.add_argument('--metric', default='AUROC', help="AUROC | Accuracy")
parser.add_argument('--ae_type', default='vae')
parser.add_argument('--latent', action='store_true')
group = parser.add_mutually_exclusive_group()
group.add_argument('--rec_error',
                   action='store_true',
                   help='model trained on the reconstruction error AE instead')
group.add_argument('--both',
                   action='store_true',
                   help='model trained on both AEs')
group.add_argument('--only_re',
                   action='store_true',
                   help='model trained only on the reconstruction error')
group.add_argument('--only_ln',
                   action='store_true',
                   help='model trained only on the latent_norm')
args = parser.parse_args()

models = ['densenet', 'resnet']
datasets = ['cifar10', 'cifar100', 'svhn']
adv_types = ['FGSM', 'BIM', 'DeepFool', 'CWL2', 'PGD100']
approaches = ['known', 'unknown']


def average_result(results: dict, contains=None):
    to_average = []
    for k, v in results.items():
        if contains is not None:
            if f'_{contains}' in k:
                to_average.append(v)
        else:
            to_average.append(v)
    return sum(to_average) / (len(to_average) + 1e-10)


def main():
    print(args, file=sys.stderr)
    name_prefix = 'latent_' if args.latent else ''
    name_prefix += 'rec_error_' if args.rec_error else ''
    name_prefix += 'both_' if args.both else ''
    name_prefix += 'ablation_re_' if args.only_re else ''
    name_prefix += 'ablation_ln_' if args.only_ln else ''
    unsupervised = True if args.classifier_type in ['IF', 'OCSVM'] else False
    # read in results
    results = {}
    results_stderr = {}
    if unsupervised:
        for model in models:
            for dataset in datasets:
                for file in args.runs_dir.iterdir():
                    pattern = f'{name_prefix}{dataset}_{model}_deep_{args.ae_type}_(.*)_{args.classifier_type}\.txt'
                    if re.match(pattern, file.name):
                        file_name = file.name
                        break
                else:
                    print(f'args.classifier_type: {args.classifier_type}',
                          file=sys.stderr)
                    for file in args.runs_dir.iterdir():
                        if re.match(f'{name_prefix}{dataset}.*', file.name):
                            print(file.name, file=sys.stderr)
                    raise FileNotFoundError(
                        f'Missing results file for: {model}/{dataset}/{args.ae_type}'
                    )
                file = args.runs_dir / file_name
                assert file.exists()
                file_contents = file.read_text()
                # print(f'file_contents: {file_contents}', file=sys.stderr)
                for adv_type in adv_types:
                    pattern = f'{args.metric} on {adv_type}: (.*?) \+/- (.*?)\n'
                    # print(f'pattern: {pattern}', file=sys.stderr)
                    matches = re.search(pattern, file_contents)
                    results[f'{model}_{dataset}_{adv_type}'] = float(
                        matches.group(1))
                    results_stderr[f'{model}_{dataset}_{adv_type}'] = float(
                        matches.group(2))
    else:
        for approach in approaches:
            for model in models:
                for dataset in datasets:
                    for file in args.runs_dir.iterdir():
                        pattern = f'{name_prefix}{dataset}_{model}_deep_{args.ae_type}_(.*)_{args.classifier_type}\.txt'
                        if re.match(pattern, file.name):
                            file_name = file.name
                            break
                    else:
                        print(
                            f'approach: {approach} args.classifier_type: {args.classifier_type}',
                            file=sys.stderr)
                        for file in args.runs_dir.iterdir():
                            if re.match(f'{name_prefix}{dataset}.*',
                                        file.name):
                                print(file.name, file=sys.stderr)
                        raise FileNotFoundError(
                            f'Missing results file for: {model}/{dataset}/{args.ae_type}'
                        )
                    file = args.runs_dir / file_name
                    assert file.exists()
                    file_contents = file.read_text()
                    for adv_type in adv_types:
                        matches = re.search(
                            f'{args.metric} on {adv_type}\({approach}\): (.*?) \+/- (.*?)\n',
                            file_contents)
                        key = f'{model}_{dataset}_{adv_type}_{approach}'
                        results[key] = float(matches.group(1))
                        results_stderr[key] = float(matches.group(2))
    # display a table/something
    best_results = {}
    # ==================================
    best_results['densenet_cifar10_FGSM_known'] = 0.9994
    best_results['densenet_cifar100_FGSM_known'] = 0.9986
    best_results['densenet_svhn_FGSM_known'] = 0.9985
    best_results['resnet_cifar10_FGSM_known'] = 0.9994
    best_results['resnet_cifar100_FGSM_known'] = 0.9977
    best_results['resnet_svhn_FGSM_known'] = 0.9962
    # ==================================
    best_results['densenet_cifar10_BIM_known'] = 0.9978
    best_results['densenet_cifar100_BIM_known'] = 0.9917
    best_results['densenet_svhn_BIM_known'] = 0.9928
    best_results['resnet_cifar10_BIM_known'] = 0.9957
    best_results['resnet_cifar100_BIM_known'] = 0.9690
    best_results['resnet_svhn_BIM_known'] = 0.9715
    # ==================================
    best_results['densenet_cifar10_DeepFool_known'] = 0.8514
    best_results['densenet_cifar100_DeepFool_known'] = 0.7757
    best_results['densenet_svhn_DeepFool_known'] = 0.9510
    best_results['resnet_cifar10_DeepFool_known'] = 0.9157
    best_results['resnet_cifar100_DeepFool_known'] = 0.8526
    best_results['resnet_svhn_DeepFool_known'] = 0.9573
    # ==================================
    best_results['densenet_cifar10_CWL2_known'] = 0.8731
    best_results['densenet_cifar100_CWL2_known'] = 0.8705
    best_results['densenet_svhn_CWL2_known'] = 0.9703
    best_results['resnet_cifar10_CWL2_known'] = 0.9584
    best_results['resnet_cifar100_CWL2_known'] = 0.9177
    best_results['resnet_svhn_CWL2_known'] = 0.9215
    # ==================================
    best_results['densenet_cifar10_FGSM_unknown'] = best_results[
        'densenet_cifar10_FGSM_known']
    best_results['densenet_cifar100_FGSM_unknown'] = best_results[
        'densenet_cifar100_FGSM_known']
    best_results['densenet_svhn_FGSM_unknown'] = best_results[
        'densenet_svhn_FGSM_known']
    best_results['resnet_cifar10_FGSM_unknown'] = best_results[
        'resnet_cifar10_FGSM_known']
    best_results['resnet_cifar100_FGSM_unknown'] = best_results[
        'resnet_cifar100_FGSM_known']
    best_results['resnet_svhn_FGSM_unknown'] = best_results[
        'resnet_svhn_FGSM_known']
    # ==================================
    best_results['densenet_cifar10_BIM_unknown'] = 0.9951
    best_results['densenet_cifar100_BIM_unknown'] = 0.9827
    best_results['densenet_svhn_BIM_unknown'] = 0.9912
    best_results['resnet_cifar10_BIM_unknown'] = 0.9891
    best_results['resnet_cifar100_BIM_unknown'] = 0.9638
    best_results['resnet_svhn_BIM_unknown'] = 0.9539
    # ==================================
    best_results['densenet_cifar10_DeepFool_unknown'] = 0.8342
    best_results['densenet_cifar100_DeepFool_unknown'] = 0.7563
    best_results['densenet_svhn_DeepFool_unknown'] = 0.9347
    best_results['resnet_cifar10_DeepFool_unknown'] = 0.7806
    best_results['resnet_cifar100_DeepFool_unknown'] = 0.8195
    best_results['resnet_svhn_DeepFool_unknown'] = 0.8430
    # ==================================
    best_results['densenet_cifar10_CWL2_unknown'] = 0.8795
    best_results['densenet_cifar100_CWL2_unknown'] = 0.8620
    best_results['densenet_svhn_CWL2_unknown'] = 0.9695
    best_results['resnet_cifar10_CWL2_unknown'] = 0.9390
    best_results['resnet_cifar100_CWL2_unknown'] = 0.9096
    best_results['resnet_svhn_CWL2_unknown'] = 0.8673
    # ==================================
    # ==================================
    best_results['densenet_cifar10_PGD100_known'] = 0.0
    best_results['densenet_cifar100_PGD100_known'] = 0.0
    best_results['densenet_svhn_PGD100_known'] = 0.0
    best_results['resnet_cifar10_PGD100_known'] = 0.0
    best_results['resnet_cifar100_PGD100_known'] = 0.0
    best_results['resnet_svhn_PGD100_known'] = 0.0
    # ==================================
    best_results['densenet_cifar10_PGD100_unknown'] = 0.0
    best_results['densenet_cifar100_PGD100_unknown'] = 0.0
    best_results['densenet_svhn_PGD100_unknown'] = 0.0
    best_results['resnet_cifar10_PGD100_unknown'] = 0.0
    best_results['resnet_cifar100_PGD100_unknown'] = 0.0
    best_results['resnet_svhn_PGD100_unknown'] = 0.0
    # ==================================
    rows = []
    rows.append(['model', 'dataset'] + adv_types + adv_types[1:])
    better_counter = 0
    for model in models:
        for dataset in datasets:
            row = [model, dataset]
            if unsupervised:
                for adv_type in adv_types:
                    key = f'{model}_{dataset}_{adv_type}'
                    key_compare = f'{model}_{dataset}_{adv_type}_unknown'
                    if results[key] > best_results[key_compare]:
                        better_counter += 1
                        better = '!'
                    else:
                        better = ''
                    row_content = f'{results[key]:.4f}{better} +/- {results_stderr[key]:.4f}'
                    row.append(row_content)
            else:
                for approach in approaches:
                    for adv_type in adv_types:
                        if approach == 'unknown' and adv_type == 'FGSM':
                            continue
                        key = f'{model}_{dataset}_{adv_type}_{approach}'
                        if results[key] > best_results[key]:
                            better_counter += 1
                            better = '!'
                        else:
                            better = ''
                        row_content = f'{results[key]:.4f}{better} +/- {results_stderr[key]:.4f}'
                        row.append(row_content)
            rows.append(row)
    print(tabulate(rows, headers='firstrow'))
    print(f'Better results count: {better_counter}')
    print(f'Averaged AUROC for known: {average_result(results, "known")}')
    print(f'Averaged AUROC for unknown: {average_result(results, "unknown")}')
    print(f'Averaged AUROC: {average_result(results)}')


if __name__ == '__main__':
    main()
