import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator

from fmclp.utils import _answer_creator, _zeros_ones_to_classes


def _ml_model(df: pd.DataFrame, estimator: BaseEstimator, prefit: bool):
    """Make ML model to run FMCLP.

    Prepare test/train split and sensitive attribute, fit if required initial classifier."""
    y = df.drop('target', axis=1)
    x = df['target']
    y_train, y_test, x_train, x_test = train_test_split(y, x)
    if not prefit:
        estimator.fit(y_train, x_train)
    estimator_pred = estimator.predict(y_test)
    accuracy_estimator = accuracy_score(estimator_pred, x_test)

    zero_train_features = y_train[y_train['attr'] == 0]
    one_train_features = y_train[y_train['attr'] == 1]
    zero_train_labels = x_train[zero_train_features.index]
    one_train_labels = x_train[one_train_features.index]

    zero_test_features = y_test[y_test['attr'] == 0]
    one_test_features = y_test[y_test['attr'] == 1]

    zero_total = zero_train_features.shape[0]
    one_total = one_train_features.shape[0]

    one_ratio = one_total / (one_total + zero_total)
    zero_ratio = zero_total / (one_total + zero_total)
    group = int(np.sqrt(one_total + zero_total))
    one_group = int(one_ratio * group)
    zero_group = int(zero_ratio * group)

    one_train_probs = pd.DataFrame(estimator.predict_proba(one_train_features)).rename(
        columns={0: 'zero_class', 1: 'first_class', 2: 'second_class'})
    one_train_probs['label'] = np.array(one_train_labels)

    zero_train_probs = pd.DataFrame(estimator.predict_proba(zero_train_features)).rename(
        columns={0: 'zero_class', 1: 'first_class', 2: 'second_class'})
    zero_train_probs['label'] = np.array(zero_train_labels)

    one_test_probs = pd.DataFrame(estimator.predict_proba(one_test_features)).rename(
        columns={0: 'zero_class', 1: 'first_class', 2: 'second_class'})

    zero_test_probs = pd.DataFrame(estimator.predict_proba(zero_test_features)).rename(
        columns={0: 'zero_class', 1: 'first_class', 2: 'second_class'})

    return {'estimator': estimator,
            'y_test': y_test,
            'x_test': x_test,
            'predictions': estimator_pred,
            'estimator_accuracy': accuracy_estimator,
            'one_group': one_group,
            'zero_group': zero_group,
            'one_train_probs': one_train_probs,
            'zero_train_probs': zero_train_probs,
            'one_test_probs': one_test_probs,
            'zero_test_probs': zero_test_probs}


def _lp_solver(d: dict, number_iterations: int, classifier: BaseEstimator, verbose: bool, multiplier: int):
    """Linear programming solver, allows to relabel observations to improve fairness."""
    one_group = multiplier * d['one_group']
    zero_group = multiplier * d['zero_group']

    bounds = [(0, 1)] * (3 * one_group + 3 * zero_group)

    equation_matrix0 = np.zeros((one_group + zero_group, 3 * one_group + 3 * zero_group))
    for i in range(one_group + zero_group):
        equation_matrix0[i, 3 * i] = 1
        equation_matrix0[i, 3 * i + 1] = 1
        equation_matrix0[i, 3 * i + 2] = 1
    equation_matrix0 = np.array(equation_matrix0)

    equation_vector = [1] * (one_group + zero_group) + [0, 0, 0]

    one_predictor_array = []
    zero_predictor_array = []

    if verbose:
        print('Start fitting')

    for k in range(number_iterations):

        one_sample = d['one_train_probs'].sample(one_group)
        zero_sample = d['zero_train_probs'].sample(zero_group)

        I0 = one_sample[one_sample['label'] == 0]
        I1 = one_sample[one_sample['label'] == 1]
        I2 = one_sample[one_sample['label'] == 2]

        J0 = zero_sample[zero_sample['label'] == 0]
        J1 = zero_sample[zero_sample['label'] == 1]
        J2 = zero_sample[zero_sample['label'] == 2]

        vectorI0 = []
        vectorI1 = []
        vectorI2 = []
        for i in one_sample.index:
            if i in I0.index:
                vectorI0.append(len(J0))
                vectorI0.append(0)
                vectorI0.append(0)
            else:
                vectorI0.append(0)
                vectorI0.append(0)
                vectorI0.append(0)
        for i in one_sample.index:
            if i in I1.index:
                vectorI1.append(0)
                vectorI1.append(len(J1))
                vectorI1.append(0)
            else:
                vectorI1.append(0)
                vectorI1.append(0)
                vectorI1.append(0)
        for i in one_sample.index:
            if i in I2.index:
                vectorI2.append(0)
                vectorI2.append(0)
                vectorI2.append(len(J2))
            else:
                vectorI2.append(0)
                vectorI2.append(0)
                vectorI2.append(0)
        vectorI0 = np.array(vectorI0)
        vectorI1 = np.array(vectorI1)
        vectorI2 = np.array(vectorI2)

        vectorJ0 = []
        vectorJ1 = []
        vectorJ2 = []
        for i in zero_sample.index:
            if i in J0.index:
                vectorJ0.append(-len(I0))
                vectorJ0.append(0)
                vectorJ0.append(0)
            else:
                vectorJ0.append(0)
                vectorJ0.append(0)
                vectorJ0.append(0)
        for i in zero_sample.index:
            if i in J1.index:
                vectorJ1.append(0)
                vectorJ1.append(-len(I1))
                vectorJ1.append(0)
            else:
                vectorJ1.append(0)
                vectorJ1.append(0)
                vectorJ1.append(0)
        for i in zero_sample.index:
            if i in J2.index:
                vectorJ2.append(0)
                vectorJ2.append(0)
                vectorJ2.append(-len(I2))
            else:
                vectorJ2.append(0)
                vectorJ2.append(0)
                vectorJ2.append(0)
        vectorJ0 = np.array(vectorJ0)
        vectorJ1 = np.array(vectorJ1)
        vectorJ2 = np.array(vectorJ2)

        row0 = np.concatenate((vectorI0, vectorJ0)).reshape(1, -1)
        row1 = np.concatenate((vectorI1, vectorJ1)).reshape(1, -1)
        row2 = np.concatenate((vectorI2, vectorJ2)).reshape(1, -1)
        rows = np.concatenate((row0, row1, row2), axis=0)

        equation_matrix = np.concatenate((equation_matrix0, rows), axis=0)

        C = np.array(one_sample[['zero_class', 'first_class', 'second_class']]).ravel()
        B = np.array(zero_sample[['zero_class', 'first_class', 'second_class']]).ravel()
        objective = (-1) * np.concatenate((C, B))

        array = linprog(
            c=objective, A_ub=None, b_ub=None,
            A_eq=equation_matrix,
            b_eq=equation_vector,
            bounds=bounds, method='highs-ipm', callback=None, options=None, x0=None).x

        fair_pred = _zeros_ones_to_classes(array)
        fair_pred_one = fair_pred[:one_group]
        fair_pred_zero = fair_pred[one_group:]

        # here we prepare classes to relabeling
        one_df = pd.DataFrame(one_sample, columns=['zero_class', 'first_class', 'second_class'])
        one_predictor = classifier
        one_predictor.fit(one_df, fair_pred_one)
        one_predictor_array.append(one_predictor)

        zero_df = pd.DataFrame(zero_sample, columns=['zero_class', 'first_class', 'second_class'])
        zero_predictor = classifier
        zero_predictor.fit(zero_df, fair_pred_zero)
        zero_predictor_array.append(zero_predictor)
        if verbose:
            print(k + 1)

    if verbose:
        print('Fitting is finished')

    return {'one_predictor_array': one_predictor_array,
            'zero_predictor_array': zero_predictor_array}


def _predictor(solved: dict, d: dict, verbose: bool):
    """Makes predictions provided solution of LP program and test features."""
    if verbose:
        print('Predicting in process')
    one_predictor_array = solved['one_predictor_array']
    zero_predictor_array = solved['zero_predictor_array']

    one_probs = d['one_test_probs']
    zero_probs = d['zero_test_probs']
    one_rows = one_probs.shape[0]
    zero_rows = zero_probs.shape[0]
    one_cols = len(one_predictor_array)
    zero_cols = len(zero_predictor_array)

    one_final_array = np.empty(shape=(one_cols, one_rows))
    for i in range(one_cols):
        one_final_array[i] = one_predictor_array[i].predict(one_probs)
    one_final_array = pd.DataFrame(one_final_array)

    one_final_ans = []
    for i in range(one_rows):
        one_final_ans.append(one_final_array[i].value_counts().sort_values(ascending=False).index[0])

    zero_final_array = np.empty(shape=(zero_cols, zero_rows))
    for i in range(zero_cols):
        zero_final_array[i] = zero_predictor_array[i].predict(zero_probs)
    zero_final_array = pd.DataFrame(zero_final_array)

    zero_final_ans = []
    for i in range(zero_rows):
        zero_final_ans.append(zero_final_array[i].value_counts().sort_values(ascending=False).index[0])

    preds = _answer_creator(one_final_ans, zero_final_ans, d['y_test']['attr'])

    if verbose:
        print('Predicting is finished')

    return preds
