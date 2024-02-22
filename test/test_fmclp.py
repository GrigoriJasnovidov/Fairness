from lightgbm import LGBMClassifier

from fmclp.fmclp_algorithm import fmclp
from fmclp.get_data import get_data


def test_fmclp():
    """Test for fmclp algorithm. Run test ONLY using command 'pytest' in terminal; otherwise ImportError appears."""
    dataset = get_data('lsac', dir='')
    res = fmclp(dataset=dataset,
                estimator=LGBMClassifier(verbose=-1),
                number_iterations=20,
                interior_classifier='rf',
                verbose=True,
                multiplier=30)

    assert res['accuracy_of_initial_classifier'] - res['accuracy_of_fair_classifier'] < 0.1, \
        "Accuracy loss is too big ..."
    assert res['fairness_of_initial_classifier']['diff'] > res['fairness_of_fair_classifier']['diff'], \
        "Fairness is not improved ..."
