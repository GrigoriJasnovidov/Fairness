import os
import numpy as np

from fmclp.benefit import benefit


def _write_res(res: dict,
               name: str,
               auxiliary_res: bool,
               write_benefit: bool):
    """Write results of one particular trial in separate .txt file."""

    text = f"""initial_classifier: {str(res['model']['estimator'])}
number_iterations: {res['number_iterations']}
multiplier: {res['multiplier']}
interior_classifier: {res['interior_classifier']}
    
unfair_diff: {res['fairness_of_initial_classifier']['diff']}
unfair_accuracy: {res['accuracy_of_initial_classifier']}
fair_diff: {res['fairness_of_fair_classifier']['diff']}
fair_accuracy: {res['accuracy_of_fair_classifier']}

"""

    if write_benefit:
        b = benefit(res)
        text += f"""unfair_discriminated_group_losses: {b['unfair_discriminated_losses']}
fair_discriminated_group_losses: {b['fair_discriminated_losses']}
discriminated_group_losses_improvement: {b['improvement_0']}
unfair_discriminated_downgraded: {b['unfair_downgraded']}
fair_discriminated_downgraded: {b['fair_downgraded']}
    """

    with open(f'{name}.txt', 'w') as f:
        f.write(text)

    if auxiliary_res:
        res['fairness_of_fair_classifier']['df'].to_csv(f"{name} cuae-metric-fair.csv")
        res['fairness_of_initial_classifier']['df'].to_csv(f"{name} cuae-metric-unfair.csv")


def general_results_write(results: list,
                          dataset_name: str,
                          number_experiments: int,
                          save_folder: str,
                          write_benefit: bool):
    """Write down results of experiment.

    Args:
        results - list consisting of experiment results; all test-result information is here
        dataset_name - name for file with results; should be without '.txt' at the end
        number_experiments - number experiments
        number_iterations - number iterations in FMCLP algorithm
        save_folder - folder to save results
        write_benefit - whether to write benefit."""
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    fair_accs = np.array([res['accuracy_of_fair_classifier'] for res in results])
    unfair_accs = np.array([res['accuracy_of_initial_classifier'] for res in results])
    fair_diffs = np.array([res['fairness_of_fair_classifier']['diff'] for res in results])
    unfair_diffs = np.array([res['fairness_of_initial_classifier']['diff'] for res in results])
    benefits = [benefit(res) for res in results]
    fair_downgraded = np.array([b['fair_downgraded'] for b in benefits])
    unfair_downgraded = np.array([b['unfair_downgraded'] for b in benefits])

    text = (f"""Testing over {dataset_name}

initial_classifier: {str(results[0]['model']['estimator'])}
number_iterations: {results[0]['number_iterations']}
multiplier: {results[0]['multiplier']}
interior_classifier: {results[0]['interior_classifier']}
number_experiments: {number_experiments}

fair_accuracy_mean: {fair_accs.mean()}
fair_diff_mean: {fair_diffs.mean()}
fair_accuracy_std: {fair_accs.std()}
fair_diff_std: {fair_diffs.std()}

unfair_accuracy_mean: {unfair_accs.mean()}
unfair_diff_mean: {unfair_diffs.mean()}
unfair_accuracy_std: {unfair_accs.std()}
unfair_diff_std: {unfair_diffs.std()}

fair_diff: {fair_diffs}

fair_accuracy: {fair_accs}

unfair_diff: {unfair_diffs}

unfair_accuracy: {unfair_accs}
""")

    if write_benefit:
        text += f"""fair_downgraded_mean: {fair_downgraded.mean()}
unfair_downgraded: {unfair_downgraded.mean()}

fair_downgraded: {fair_downgraded}
unfair_downgraded: {unfair_downgraded}"""

    with open(f'{save_folder}/{dataset_name}.txt', 'w') as f:
        f.write(text)
