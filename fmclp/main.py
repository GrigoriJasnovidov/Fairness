from lightgbm import LGBMClassifier

from fmclp.fmclp_algorithm import fmclp
from fmclp.get_data import get_data

if __name__ == '__main__':
    dataset = get_data('lsac')
    res = fmclp(dataset=dataset,
                estimator=LGBMClassifier(verbose=-1),
                number_iterations=20,
                interior_classifier='rf',
                verbose=True,
                multiplier=30)
    print(f"""Accuracy of initial classifier: {res['accuracy_of_initial_classifier']}
fairness of initial classifier: {res['fairness_of_initial_classifier']['diff']}
accuracy of fair classifier: {res['accuracy_of_fair_classifier']}
fairness of fair classifier {res['fairness_of_fair_classifier']['diff']}""")
