#!/usr/bin/env python
# coding: utf-8

"""The code in this file fits three models for each justice and saves out the
best models and the model results in pickled dataframes.
"""

import os
import pandas as pd
import pickle
from scipy.stats import randint as sp_randint, uniform as sp_uniform
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit
from xgboost import XGBClassifier as XGC


def read_data(fn):
    with open(fn, 'rb') as fp:
        return pickle.load(fp)


def write_data(output, fn):
    with open(fn, 'wb') as fp:
        pickle.dump(output, fp)


def get_best_model(clf, params, X_train, y_train, scoring='roc_auc',
                   n_iterations=10):
    """Run randomized grid search to optimize model"""
    cv_sets = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)

    random_search_obj = RandomizedSearchCV(estimator=clf,
                                           param_distributions=params,
                                           n_iter=n_iterations, scoring=scoring,
                                           cv=cv_sets, n_jobs=-1)

    random_search_fit = random_search_obj.fit(X_train, y_train)

    best_clf = random_search_fit.best_estimator_
    best_params = random_search_fit.best_params_
    return best_clf, best_params


def get_predictions(best_clf, clf, X_train, y_train, X_test, y_test):
    """Make predictions using the unoptimized and optimized models"""
    predictions = (clf.fit(X_train, y_train)).predict(X_test)
    best_predictions = best_clf.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    best_acc = accuracy_score(y_test, best_predictions)
    auc = roc_auc_score(y_test, predictions)
    best_auc = roc_auc_score(y_test, best_predictions)
    f1 = f1_score(y_test, predictions)
    best_f1 = f1_score(y_test, best_predictions)
    return acc, best_acc, auc, best_auc, f1, best_f1


def build_models(model_data_suffix, best_models_fn, model_results_fn,
                 best_results_fn, base_path):
    """Control function for the module. Sets up parameters for modeling, defines
    different subsets of the data to use in building models, and loops through
    the justices, building each of the prescribed models for each justice.

    The function also writes out all the model results as well as the models
    themselves.
    """
    structured = ['adj_petitioner', 'adj_respondent', 'issueArea',
                  'circuit', 'adj_disposition', 'certReason']
    text_metrics = ['p_duration_pct', 'p_scaled_cutoff',
                    'r_duration_pct', 'r_scaled_cutoff']
    advocate_vecs = ['fr_p_tv_' + str(i) for i in range(1, 101)] + \
                    ['fr_r_tv_' + str(i) for i in range(1, 101)]
    sc_vecs = ['to_p_tv_' + str(i) for i in range(1, 101)] + \
              ['to_r_tv_' + str(i) for i in range(1, 101)]

    # These are the three types of models per justice.
    SUBSETS = {'no_text': structured,
               'all_but_sc': structured + text_metrics + advocate_vecs,
               'all': structured + text_metrics + advocate_vecs + sc_vecs}

    JUSTICES = ['WJBrennan', 'BRWhite', 'TMarshall', 'HABlackmun',
                'WHRehnquist', 'JPStevens', 'SDOConnor', 'AScalia', 'AMKennedy',
                'CThomas', 'RBGinsburg', 'SGBreyer']

    CLF = XGC(random_state=17)

    # Grid search parameters.
    PARAMS = {'max_depth': sp_randint(low=5, high=16),
              'learning_rate': sp_uniform(loc=0.01, scale=0.19),
              'n_estimators': sp_randint(low=200, high=601),
              'gamma': sp_uniform(loc=0.0, scale=4.0),
              'subsample': sp_uniform(loc=0.5, scale=0.5),
              'colsample_bytree': sp_uniform(loc=0.3, scale=0.7)}

    records = []
    best_models = {}

    for _justice in JUSTICES:
        model_data_fn = os.path.join(base_path, _justice + model_data_suffix)
        X_train, X_test, y_train, y_test = read_data(model_data_fn)
        record = {'justice': _justice}
        record['num_cases'] = len(y_train) + len(y_test)
        record['baseline'] = (y_train.sum() + y_test.sum()) / record['num_cases']
        for _subset in SUBSETS:
            X_trn = X_train[SUBSETS[_subset]]
            X_tst = X_test[SUBSETS[_subset]]
            best_clf, best_params = get_best_model(CLF, PARAMS, X_trn, y_train,
                                                   n_iterations=20)
            acc, best_acc, auc, best_auc, f1, best_f1 = \
                get_predictions(best_clf, CLF, X_trn, y_train, X_tst, y_test)
            best_models[_justice + '_' + _subset] = best_clf
            record[_subset.upper() + '_best_params'] = best_params
            record[_subset.upper() + '_acc'] = acc
            record[_subset.upper() + '_best_acc'] = best_acc
            record[_subset.upper() + '_auc'] = auc
            record[_subset.upper() + '_best_auc'] = best_auc
            record[_subset.upper() + '_f1'] = f1
            record[_subset.upper() + '_best_f1'] = best_f1
        records.append(record)

    model_results = pd.DataFrame(records)

    write_data(best_models, best_models_fn)
    write_data(model_results, model_results_fn)

    best_cols = ['justice', 'baseline', 'num_cases'] + \
        [c for c in model_results.columns if ('params' not in c and 'best' in c)]
    write_data(model_results[best_cols], best_results_fn)
    return


def run(model_data_suffix, best_models, model_results,
        best_results, base):
    base_path = os.path.expanduser(base)
    best_models_fn = os.join.path(base_path, best_models)
    model_results_fn = os.join.path(base_path, model_results)
    best_results_fn = os.join.path(base_path, best_results)
    build_models(model_data_suffix, best_models_fn, model_results_fn,
                 best_results_fn, base_path)


if __name__ == '__main__':
    run('_model_data.p', 'best_models.p', 'model_results.p',
        'best_results.p', base='~/projects/insight/__data__/SCOTUS')
