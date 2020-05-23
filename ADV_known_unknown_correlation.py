import argparse
import re
from pathlib import Path

import joblib
import numpy as np
from joblib import parallel_backend
from sklearn.ensemble import (GradientBoostingClassifier, IsolationForest,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC, OneClassSVM

from scorings import score_func
import traceback
import pickle


def find_dir_for(parent, model, dataset):
    pattern = f'{dataset}_{model}_deep_(.*)_.*'
    for file in parent.iterdir():
        if file.is_dir():
            if re.match(pattern, file.name):
                return file


def get_dirs_for(parent, model, dataset):
    dirs = []
    pattern = f'{dataset}_{model}_deep_(.*)_.*'
    for file in parent.iterdir():
        if file.is_dir():
            if re.match(pattern, file.name):
                # print(f'Including {file}')
                dirs.append(file)
    return dirs


def get_results_for(dir, adv_type, known_adv_type, metric):
    results = []
    pattern_known = f'.*?results_(.*?)_unknown\.txt'
    for file in dir.iterdir():
        match = re.match(pattern_known, file.name)
        if match:
            # read in result from the corresponding "unknown" file
            try:
                file_contents = file.read_text()
                result_matches = re.search(f'{metric} on {adv_type}: (.*?)\n',
                                           file_contents)
                unknown_result = float(result_matches.group(1))
            except:
                # print(
                #     f'Error when reading {str(file.name)} contents for {adv_type} results.'
                # )
                # traceback.print_exc()
                continue
            # read in result from this file
            classifier_str = match.group(1)
            known_file = file.parent / f'results_{classifier_str}_known.txt'
            try:
                assert known_file.exists()
                file_contents = known_file.read_text()
                result_matches = re.search(
                    f'{metric} on {known_adv_type}: (.*?)\n', file_contents)
                known_result = float(result_matches.group(1))
            except:
                # print(
                #     f'Error when reading {str(file.name)} contents for {known_adv_type} results.'
                # )
                # traceback.print_exc()
                continue

            # results.append((classifier_str, known_result, unknown_result))
            results.append((known_result, unknown_result))
    return results


def compute_correlation(x_list):
    x = np.array(x_list)
    assert x.shape[1] == 2
    return np.corrcoef(x[:, 0], x[:, 1])[1][0]


def main():
    pipeline = Pipeline([
        ('std', None),
        ('clf', None),
    ])
    # TODO extract parameters to a common file
    if args.model == 'SVC':
        params = [
            {
                'std': [MinMaxScaler(), StandardScaler(), None],
                'clf': [SVC()],
                'clf__kernel': ['rbf', 'poly'],
                'clf__C': [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                'clf__gamma': ['scale'],
            },
        ]
    elif args.model == 'LSVC':
        params = [
            {
                'std': [MinMaxScaler(), StandardScaler(), None],
                'clf': [LinearSVC()],
                'clf__C': [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                'clf__max_iter': [100000],
            },
        ]
    elif args.model == 'LR':
        params = [
            {
                'std': [MinMaxScaler(), StandardScaler(), None],
                'clf': [LogisticRegression()],
            },
        ]
    elif args.model == 'RF':
        params = [
            {
                'std': [MinMaxScaler(), StandardScaler(), None],
                'clf': [RandomForestClassifier()],
                'clf__n_estimators': [
                    500,
                ],
                'clf__max_depth': [None, 2, 8, 16],
                'clf__min_samples_split': [2, 0.1, 0.5],
                'clf__max_features': ['sqrt', 'log2'],
            },
        ]
    elif args.model == 'GB':
        params = [
            {
                'std': [MinMaxScaler(), StandardScaler(), None],
                'clf': [GradientBoostingClassifier()],
                'clf__loss': ['deviance', 'exponential'],
                'clf__learning_rate': [0.5, 0.1, 0.01, 0.001],
                'clf__n_estimators': [32, 100, 200, 500],
                'clf__max_depth': [2, 4, 8, 16],
                'clf__min_samples_split': [2, 0.1, 0.5],
            },
        ]
    name_prefix = 'latent_' if args.latent else ''

    main_dir = args.main_dir

    models = ['resnet', 'densenet']
    datasets = ['cifar10', 'cifar100', 'svhn']
    adv_types = ['FGSM', 'BIM', 'DeepFool', 'CWL2', 'PGD100']
    adv_known = args.adv_known

    results_aggr = []
    for adv_type in set(adv_types) - {adv_known}:
        results_aggr_adv = []
        for dataset in datasets:
            for model in models:
                results = []
                dirs = get_dirs_for(Path(main_dir), model, dataset)
                for d in dirs:
                    res = get_results_for(d, adv_type, adv_known, 'AUROC')
                    results.extend(res)
                    results_aggr_adv.extend(res)
                    results_aggr.extend(res)
                # if len(results) > 0:
                #     print(
                #         f'Correlation ({len(results)} results) for {model} {dataset} {adv_type}: {compute_correlation(results)}'
                #     )
                # else:
                #     print(f'Empty for {model} {dataset} {adv_type} ')
            # print(
            #     f'Correlation for {model} {dataset}: {compute_correlation(results_aggr_adv)}'
            # )
        print(
            f'Correlation for {adv_type} ({len(results_aggr_adv)} results): {compute_correlation(results_aggr_adv)}'
        )
    print(
        f'Correlation ({len(results_aggr)} results): {compute_correlation(results_aggr)}'
    )

    train_data_sizes = [0.0005] + [(i + 1) * 0.001 for i in range(9)] + [(i + 1) * 0.01 for i in range(10)]
    runs = 10

    for model in ['resnet', 'densenet']:
        for dataset in ['cifar10', 'cifar100', 'svhn']:
            dataset_names = ['train']
            for adv_type in adv_types:
                dataset_names.append(f'clean_{adv_type}')
                dataset_names.append(f'adv_{adv_type}')
                dataset_names.append(f'noisy_{adv_type}')

            run_dir = find_dir_for(Path(main_dir), model, dataset)
            print(f'Choosing directory: {str(run_dir)}')
            datasets = {}
            for name in dataset_names:
                if args.latent:
                    dataset_path = run_dir / f'latent_{name}.npy'
                else:
                    dataset_path = run_dir / f'ae_encoded_{name}.npy'
                if isinstance(dataset_path, Path):
                    if dataset_path.exists():
                        datasets[name] = np.load(str(dataset_path))
                    else:
                        print(f'{dataset_path} is missing!')
                else:
                    datasets[name] = []
                    for path in dataset_path:
                        assert path.exists(), f'{path} is missing!'
                        datasets[name].append(np.load(str(path)))
                    datasets[name] = np.concatenate(datasets[name], axis=1)

            # "known" part
            cache_file = Path(main_dir) / f'{name_prefix}known_{adv_known}_unknown_{args.model}_{dataset}_{model}.pickle'
            if cache_file.exists():
                with open(cache_file, 'rb') as cf:
                    results = pickle.load(cf)
            else:
                results = {}

            for train_split in train_data_sizes:
                for run_n in range(runs):
                    key = f'{train_split}_{run_n}'
                    if key not in results:
                        # train on adv_known
                        train_size = int(train_split *
                                        len(datasets[f'clean_{adv_types[0]}']))
                        test_size = len(datasets[f'clean_{adv_types[0]}']) - train_size
                        X = np.concatenate([
                            datasets[f'clean_{adv_types[0]}'][:train_size],
                            datasets[f'adv_{adv_types[0]}'][:train_size],
                            datasets[f'noisy_{adv_types[0]}'][:train_size],
                        ])
                        y = np.concatenate([
                            np.ones(train_size),
                            np.zeros(train_size),
                            np.ones(train_size),
                        ])
                        X_test = np.concatenate([
                            datasets[f'clean_{adv_types[0]}'][train_size:],
                            datasets[f'adv_{adv_types[0]}'][train_size:],
                            datasets[f'noisy_{adv_types[0]}'][train_size:],
                        ])
                        y_test = np.concatenate([
                            np.ones(test_size),
                            np.zeros(test_size),
                            np.ones(test_size),
                        ])
                        with parallel_backend('loky', n_jobs=args.jobs):
                            gs = GridSearchCV(pipeline,
                                            params,
                                            scoring=make_scorer(
                                                roc_auc_score, needs_threshold=True),
                                            #   cv=StratifiedKFold(5),
                                            cv=StratifiedKFold(2),
                                            verbose=1)
                            gs.fit(X, y)
                        # evaluate on adv_known
                        y_pred = gs.predict(X_test)
                        try:
                            y_scores = gs.decision_function(X_test)
                        except:
                            y_scores = gs.predict_proba(X_test)
                            if y_scores.ndim > 1:
                                y_scores = y_scores[:, 1]
                        adv_known_auroc = roc_auc_score(y_test, y_scores)
                        # evaluate on other attacks
                        adv_aurocs = {}
                        for adv_type in set(adv_types) - {adv_known}:
                            test_size = len(datasets[f'clean_{adv_type}'])
                            X_test = np.concatenate([
                                datasets[f'clean_{adv_type}'],
                                datasets[f'adv_{adv_type}'],
                                datasets[f'noisy_{adv_type}'],
                            ])
                            y_test = np.concatenate([
                                np.ones(test_size),
                                np.zeros(test_size),
                                np.ones(test_size),
                            ])
                            y_pred = gs.predict(X_test)
                            try:
                                y_scores = gs.decision_function(X_test)
                            except:
                                y_scores = gs.predict_proba(X_test)
                                if y_scores.ndim > 1:
                                    y_scores = y_scores[:, 1]
                            adv_aurocs[adv_type] = roc_auc_score(y_test, y_scores)
                        # results[key] = (adv_known_auroc, *[adv_aurocs[t] for t in adv_types])
                        results[key] = (adv_known_auroc, adv_aurocs)
                        with open(cache_file, 'wb') as cf:
                            pickle.dump(results, cf)
            
            # TODO plot something instead
            for adv_type in set(adv_types) - {adv_known}:
                print(f'======== {adv_type} on {model} {dataset} ========')
                print(f'SPLIT: {adv_known} {adv_type}')
                results_aggr_adv = []
                for train_split in train_data_sizes:
                    split_results = []
                    for run_n in range(runs):
                        key = f'{train_split}_{run_n}'
                        res = (results[key][0], results[key][1][adv_type])
                        split_results.append(res)
                        results_aggr_adv.append(res)
                    averaged_results = np.array(split_results).mean(axis=0)
                    print(f'{train_split}: {averaged_results[0]} {averaged_results[1]}')
                print(f'Correlation ({len(results_aggr_adv)}): {compute_correlation(results_aggr_adv)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('main_dir', type=str)
    parser.add_argument('--model',
                        default='SVC',
                        help='LR | SVC | OneClassSVM | IsolationForest')
    parser.add_argument('--latent',
                        action='store_true',
                        help='train model on the whole latent representation')
    parser.add_argument('--jobs',
                        default=20,
                        type=int,
                        help='number of joblib jobs')
    parser.add_argument('--adv_known',
                        default='FGSM')
    args = parser.parse_args()
    print(args)
    main()
