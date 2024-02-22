import os
import random
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.base import BaseEstimator

from fmclp.fmclp_algorithm import fmclp
from fmclp.write_results import _write_res, general_results_write


def run_experiment(dataset: pd.DataFrame,
                   dataset_name: str,
                   folder: str,
                   number_experiments: int,
                   multiplier: int,
                   number_iterations: int,
                   interior_classifier: str,
                   initial_classifier: BaseEstimator = LGBMClassifier(verbose=-1),
                   write_each_trial: bool = False,
                   write_benefit: bool = False,
                   write_auxiliary: bool = False):
    """Run experiment and write down results.

    Args:
        dataset - dataframe with mandatory 'attr' and 'target' columns
        dataset_name - file name to write down results without '.txt' at the end
        folder - folder to write results
        number_experiments - number experiments
        multiplier - multiplier parameter for FMCLP algorithm
        number_iterations - number iterations parameter for FMCLP algorithm
        interior_classifier - interior_classifier parameter for FMCLP algorithm
        initial_classifier - initial (unfair) classifier
        write_each_trial - whether to write each trial in a separate file
        write_benefit - whether to compute and write benefit
        write_auxiliary - whether to write .csv files with cuae-metric for each file.
    Returns: list of results for all trials."""
    if not os.path.exists(folder):
        os.mkdir(folder)
    results = []

    for i in range(number_experiments):
        res = fmclp(dataset=dataset,
                    estimator=initial_classifier,
                    number_iterations=number_iterations,
                    prefit=False,
                    interior_classifier=interior_classifier,
                    verbose=False,
                    multiplier=multiplier)
        results.append(res)
        if write_each_trial:
            _write_res(res=res,
                       name=f"{folder}/{dataset_name}_№{i + 1}",
                       auxiliary_res=write_auxiliary,
                       write_benefit=write_benefit)
        print(i + 1)

    general_results_write(results=results,
                          dataset_name=dataset_name,
                          number_experiments=number_experiments,
                          save_folder=folder,
                          write_benefit=write_benefit)

    return results
