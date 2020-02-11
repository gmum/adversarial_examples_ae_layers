import argparse
from pathlib import Path

import joblib
import numpy as np
from joblib import parallel_backend
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import OneClassSVM, SVC, LinearSVC

from scorings import score_func

adv_types = ['FGSM', 'BIM', 'DeepFool', 'CWL2']

N_JOBS = 20


def main():
    unsupervised_models = ['OCSVM', 'IF']
    is_supervised = args.model not in unsupervised_models
    # outliers = args.outliers / 100
    outliers = 0.05
    in_class = 1.0 - outliers

    pipeline = Pipeline([
        ('std', None),
        ('clf', None),
    ])
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
                'clf__n_estimators': [500, ],
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
    elif args.model == 'IF':
        params = [
            {
                'std': [MinMaxScaler(), StandardScaler(), None],
                'clf': [IsolationForest()],
                'clf__n_estimators': [20, 50, 100, 200],
                'clf__contamination': [outliers, outliers + 0.025, outliers - 0.025],
                'clf__max_samples': ['auto', 0.1],
                'clf__bootstrap': [True, False],
                'clf__behaviour': [True],
            },
        ]
    elif args.model == 'OCSVM':
        params = [
            {
                'std': [MinMaxScaler(), StandardScaler(), None],
                'clf': [OneClassSVM()],
                'clf__kernel': ['rbf', 'poly', 'linear'],
                'clf__gamma': ['scale', 'auto'],
                'clf__nu': [outliers / 2, outliers, outliers * 2],
            },
        ]

    run_dir_name = args.run_dir
    latent_prefix = 'latent_' if args.latent else ''

    dataset_names = ['train']
    for adv_type in adv_types:
        dataset_names.append(f'clean_{adv_type}')
        dataset_names.append(f'adv_{adv_type}')
        dataset_names.append(f'noisy_{adv_type}')

    # TODO results dict, aggregate and print mean and stddev to a new file
    results = {}
    for run_n in range(args.runs):
        results[run_n] = {}
        run_dir = Path(f'{run_dir_name}_{run_n}')
        run_name = str(run_dir)
        assert run_dir.exists()

        datasets = {}
        for name in dataset_names:
            if args.latent:
                dataset_path = run_dir / f'latent_{name}.npy'
            else:
                dataset_path = run_dir / f'ae_encoded_{name}.npy'
            if dataset_path.exists():
                datasets[name] = np.load(str(dataset_path))
            else:
                print(f'{dataset_path} is missing!')

        # for supervised we consider two setups - "known attack" (left half of table 3 from "A Simple Unified Framework...")
        # and "unknown attack" where we train only on FGSM and validate on the rest
        # for unsupervised we train on entire training data(! - change if needed) and test on clean/adv/noisy
        if is_supervised:
            # "known" part
            results_filename = f'{run_name}/{latent_prefix}results_{args.model}_known.txt'
            if not Path(results_filename).exists():
                with open(results_filename, 'x') as results_file:
                    for adv_type in adv_types:
                        model_filename = f'{run_name}/{latent_prefix}final_cv_{args.model}_known_{adv_type}.joblib'
                        train_split = 0.1
                        train_size = int(train_split * len(datasets[f'clean_{adv_type}']))
                        test_size = len(datasets[f'clean_{adv_type}']) - train_size
                        X = np.concatenate([
                            datasets[f'clean_{adv_type}'][:train_size],
                            datasets[f'adv_{adv_type}'][:train_size],
                            datasets[f'noisy_{adv_type}'][:train_size],
                        ])
                        y = np.concatenate([
                            np.ones(train_size),
                            np.zeros(train_size),
                            np.ones(train_size),
                        ])
                        X_test = np.concatenate([
                            datasets[f'clean_{adv_type}'][train_size:],
                            datasets[f'adv_{adv_type}'][train_size:],
                            datasets[f'noisy_{adv_type}'][train_size:],
                        ])
                        y_test = np.concatenate([
                            np.ones(test_size),
                            np.zeros(test_size),
                            np.ones(test_size),
                        ])
                        # train
                        with parallel_backend('loky', n_jobs=N_JOBS):
                            gs = GridSearchCV(pipeline, params,
                                              scoring=make_scorer(roc_auc_score, needs_threshold=True),
                                              cv=StratifiedKFold(5), verbose=1)
                            gs.fit(X, y)
                        # save model
                        joblib.dump(gs, model_filename)
                        print(f'Best params on {adv_type}: {gs.best_params_}', file=results_file)
                        # print feature importance on Random Forest
                        if args.model == 'RF':
                            rf = gs.best_estimator_['clf']
                            print(f'RF feature importance for {adv_type}: \n {rf.feature_importances_.tolist()}',
                                  file=results_file)
                        # validate
                        y_pred = gs.predict(X_test)
                        try:
                            y_scores = gs.decision_function(X_test)
                        except:
                            y_scores = gs.predict_proba(X_test)
                            if y_scores.ndim > 1:
                                y_scores = y_scores[:, 1]
                        acc = accuracy_score(y_test, y_pred)
                        auroc = roc_auc_score(y_test, y_scores)
                        print(f'Accuracy on {adv_type}: {acc}', file=results_file)
                        results[run_n][f'acc_known_{adv_type}'] = acc
                        print(f'AUROC on {adv_type}: {auroc}', file=results_file)
                        results[run_n][f'auroc_known_{adv_type}'] = auroc
            # "unknown/FGSM" part
            results_filename = f'{run_name}/{latent_prefix}results_{args.model}_unknown.txt'
            if not Path(results_filename).exists():
                with open(results_filename, 'x') as results_file:
                    model_filename = f'{run_name}/{latent_prefix}final_cv_{args.model}_unknown.joblib'
                    # train on FGSM
                    train_split = 0.1
                    train_size = int(train_split * len(datasets[f'clean_{adv_types[0]}']))
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
                    # train
                    with parallel_backend('loky', n_jobs=N_JOBS):
                        gs = GridSearchCV(pipeline, params, scoring=make_scorer(roc_auc_score, needs_threshold=True),
                                          cv=StratifiedKFold(5), verbose=1)
                        gs.fit(X, y)
                    # save model
                    print(f'Best params: {gs.best_params_}', file=results_file)
                    joblib.dump(gs, model_filename)
                    # print feature importance on Random Forest
                    if args.model == 'RF':
                        rf = gs.best_estimator_['clf']
                        print(f'RF feature importance: \n {rf.feature_importances_.tolist()}',
                              file=results_file)
                    # test
                    y_pred = gs.predict(X_test)
                    try:
                        y_scores = gs.decision_function(X_test)
                    except:
                        y_scores = gs.predict_proba(X_test)
                        if y_scores.ndim > 1:
                            y_scores = y_scores[:, 1]
                    acc = accuracy_score(y_test, y_pred)
                    auroc = roc_auc_score(y_test, y_scores)
                    print(f'Accuracy on {adv_types[0]}: {acc}', file=results_file)
                    results[run_n][f'acc_unknown_{adv_types[0]}'] = acc
                    print(f'AUROC on {adv_types[0]}: {auroc}', file=results_file)
                    results[run_n][f'auroc_unknown_{adv_types[0]}'] = auroc
                    # and test on the rest
                    for adv_type in adv_types[1:]:
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
                        # validate
                        y_pred = gs.predict(X_test)
                        try:
                            y_scores = gs.decision_function(X_test)
                        except:
                            y_scores = gs.predict_proba(X_test)
                            if y_scores.ndim > 1:
                                y_scores = y_scores[:, 1]
                        acc = accuracy_score(y_test, y_pred)
                        auroc = roc_auc_score(y_test, y_scores)
                        print(f'Accuracy on {adv_type}: {acc}', file=results_file)
                        results[run_n][f'acc_unknown_{adv_type}'] = acc
                        print(f'AUROC on {adv_type}: {auroc}', file=results_file)
                        results[run_n][f'auroc_unknown_{adv_type}'] = auroc
        else:
            model_filename = f'{run_name}/{latent_prefix}final_cv_{args.model}.joblib'
            results_filename = f'{run_name}/{latent_prefix}results_{args.model}.txt'
            if not Path(model_filename).exists():
                # use only train dataset for one-class classifiers
                X = datasets[f'train']
                train_size = len(X)
                y = np.ones(train_size)
                with parallel_backend('loky', n_jobs=N_JOBS):
                    gs = GridSearchCV(pipeline, params, scoring=make_scorer(score_func, greater_is_better=False), cv=5,
                                      verbose=1)
                    gs.fit(X, y)
                # save model
                joblib.dump(gs, model_filename)
                # save results
                with open(results_filename, 'w') as results_file:
                    print(f'Best score: {gs.best_score_}', file=results_file)
                    print(f'Best params: {gs.best_params_}', file=results_file)
                    for adv_type in adv_types:
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
                            y_scores = gs.predict_proba(X_test)[0]
                        acc = accuracy_score(y_test, y_pred)
                        auroc = roc_auc_score(y_test, y_scores)
                        print(f'Accuracy on {adv_type}: {acc}', file=results_file)
                        results[run_n][f'acc_{adv_type}'] = acc
                        print(f'AUROC on {adv_type}: {auroc}', file=results_file)
                        results[run_n][f'auroc_{adv_type}'] = auroc
    results_filename = f'{latent_prefix}{run_dir_name}_{args.model}.txt'
    with open(results_filename, 'x') as results_file:
        for adv_type in adv_types:
            if is_supervised:
                # known
                res = np.array([results[i][f'acc_known_{adv_type}'] for i in range(args.runs)])
                print(f'Acc on {adv_type}(known): {res.mean()} +/- {res.std(ddof=1)}', file=results_file)
                res = np.array([results[i][f'auroc_known_{adv_type}'] for i in range(args.runs)])
                print(f'AUROC on {adv_type}(known): {res.mean()} +/- {res.std(ddof=1)}', file=results_file)
                # unknown
                res = np.array([results[i][f'acc_unknown_{adv_type}'] for i in range(args.runs)])
                print(f'Acc on {adv_type}(unknown): {res.mean()} +/- {res.std(ddof=1)}', file=results_file)
                res = np.array([results[i][f'auroc_unknown_{adv_type}'] for i in range(args.runs)])
                print(f'AUROC on {adv_type}(unknown): {res.mean()} +/- {res.std(ddof=1)}', file=results_file)
            else:
                res = np.array([results[i][f'acc_{adv_type}'] for i in range(args.runs)])
                print(f'Acc on {adv_type}: {res.mean()} +/- {res.std(ddof=1)}', file=results_file)
                res = np.array([results[i][f'auroc_{adv_type}'] for i in range(args.runs)])
                print(f'AUROC on {adv_type}: {res.mean()} +/- {res.std(ddof=1)}', file=results_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=str)
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--model', default='SVC', help='LR | SVC | OneClassSVM | IsolationForest')
    parser.add_argument('--latent', action='store_true', help='Train model on the whole latent representation')
    # parser.add_argument('--outliers', type=int, default=5, help='desired proportion (percent) of outliers in the trainset')
    parser.add_argument('--runs', default=5, help='number of runs')
    args = parser.parse_args()
    print(args)
    main()
