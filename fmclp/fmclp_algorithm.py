import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator
from lightgbm import LGBMClassifier

from fmclp.core import _lp_solver, _predictor, _ml_model
from fmclp.cuae_metric import cuae


def fmclp(dataset: pd.DataFrame,
          estimator: BaseEstimator = LGBMClassifier(verbose=-1),
          number_iterations: int = 10,
          prefit: bool = False,
          interior_classifier: str = 'rf',
          verbose: bool = False,
          multiplier: int = 1):
    """Run FMCLP algorithm.

    Args:
        dataset - dataframe with mandatory 'target' and 'attr' columns
        estimator - initial classifier; must support sklearn fit/predict interface
        number_iterations - number_iterations parameter of FMCLP
        prefit - if set True then estimator must be fitted and will not be refitted during workflow
        interior_classifier - interior classifier parameter; options: 'rf' - RandomForestClassifier, 'lr' -
                              LogisticRegression, 'dt' - DecisionTreeClassifier, 'svm' - SVC, 'lgb' - LGBMClassifier,
                              'knn' - KNeighborClassifier
        verbose - whether show progress
        multiplier - multiplier parameter of FMCLP algorithm.
    Returns:
        dictionary with results of implementation of FMCLP over dataset."""
    interior_classifier_dict = {'rf': RandomForestClassifier(),
                                'lr': LogisticRegression(),
                                'dt': DecisionTreeClassifier(),
                                'svm': SVC(),
                                'lgb': LGBMClassifier(verbose=-1),
                                'knn': KNeighborsClassifier(n_neighbors=3)}

    model = _ml_model(df=dataset, estimator=estimator, prefit=prefit)
    solved = _lp_solver(model,
                        classifier=interior_classifier_dict[interior_classifier],
                        number_iterations=number_iterations,
                        verbose=verbose,
                        multiplier=multiplier)
    pred = _predictor(solved, model, verbose)
    fair_cuae = cuae(y_true=model['x_test'], y_pred=pred, sensitive_features=model['y_test']['attr'])
    unfair_cuae = cuae(y_true=model['x_test'], y_pred=model['predictions'], sensitive_features=model['y_test']['attr'])
    fair_accuracy = accuracy_score(pred, model['x_test'])

    return {'accuracy_of_initial_classifier': model['estimator_accuracy'],
            'fairness_of_initial_classifier': unfair_cuae,
            'accuracy_of_fair_classifier': fair_accuracy,
            'fairness_of_fair_classifier': fair_cuae,
            'predictions': pred,
            'dataset': dataset,
            'model': model,
            'number_iterations': number_iterations,
            'multiplier': multiplier,
            'interior_classifier': interior_classifier}
