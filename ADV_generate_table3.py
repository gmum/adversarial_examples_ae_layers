import argparse
import re
from pathlib import Path

from tabulate import tabulate

parser = argparse.ArgumentParser()
parser.add_argument('runs_dir', type=Path)
parser.add_argument('--classifier_type', default='SVC')
parser.add_argument('--metric', default='AUROC', help="AUROC | Accuracy")
parser.add_argument('--ae_type', default='vae')
parser.add_argument('--latent', action='store_true')
args = parser.parse_args()

models = ['densenet', 'resnet']
datasets = ['cifar10', 'cifar100', 'svhn']
adv_types = ['FGSM', 'BIM', 'DeepFool', 'CWL2']
approaches = ['known', 'unknown']


def main():
    latent_prefix = 'latent_' if args.latent else ''
    unsupervised = True if args.classifier_type in ['IF', 'OCSVM'] else False
    # read in results
    results = {}
    if unsupervised:
        for model in models:
            for dataset in datasets:
                for file in args.runs_dir.iterdir():
                    if re.match(f'{dataset}_{model}_deep_{args.ae_type}_(.*)', file.name):
                        dir_name = file.name
                        break
                else:
                    raise FileNotFoundError(f'Missing results file for: {model}/{dataset}/{args.ae_type}')
                file_name = f'{latent_prefix}results_{args.classifier_type}.txt'
                file = args.runs_dir / dir_name / file_name
                file_contents = file.read_text()
                for adv_type in adv_types:
                    result = float(re.search(f'{args.metric} on {adv_type}: (.*)', file_contents)[1])
                    results[f'{model}_{dataset}_{adv_type}'] = result
    else:
        for approach in approaches:
            for model in models:
                for dataset in datasets:
                    for file in args.runs_dir.iterdir():
                        if re.match(f'{dataset}_{model}_deep_{args.ae_type}_(.*)', file.name):
                            dir_name = file.name
                            break
                    else:
                        raise FileNotFoundError(f'Missing results file for: {model}/{dataset}/{args.ae_type}')
                    file_name = f'{latent_prefix}results_{args.classifier_type}_{approach}.txt'
                    file = args.runs_dir / dir_name / file_name
                    file_contents = file.read_text()
                    for adv_type in adv_types:
                        result = float(re.search(f'{args.metric} on {adv_type}: (.*)', file_contents)[1])
                        results[f'{model}_{dataset}_{adv_type}_{approach}'] = result

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
    best_results['densenet_cifar10_FGSM_unknown'] = best_results['densenet_cifar10_FGSM_known']
    best_results['densenet_cifar100_FGSM_unknown'] = best_results['densenet_cifar100_FGSM_known']
    best_results['densenet_svhn_FGSM_unknown'] = best_results['densenet_svhn_FGSM_known']
    best_results['resnet_cifar10_FGSM_unknown'] = best_results['resnet_cifar10_FGSM_known']
    best_results['resnet_cifar100_FGSM_unknown'] = best_results['resnet_cifar100_FGSM_known']
    best_results['resnet_svhn_FGSM_unknown'] = best_results['resnet_svhn_FGSM_known']
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
                        row.append(f'{results[key]:.4f}!')
                    else:
                        row.append(f'{results[key]:.4f}')
            else:
                for approach in approaches:
                    for adv_type in adv_types:
                        if approach == 'unknown' and adv_type == 'FGSM':
                            continue
                        key = f'{model}_{dataset}_{adv_type}_{approach}'
                        if results[key] > best_results[key]:
                            better_counter += 1
                            row.append(f'{results[key]:.4f}!')
                        else:
                            row.append(f'{results[key]:.4f}')
            rows.append(row)
    print(tabulate(rows, headers='firstrow'))
    print(f'Better results count: {better_counter}')


if __name__ == '__main__':
    main()
