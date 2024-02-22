import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


def cuae(y_true: pd.Series, y_pred: pd.Series, sensitive_features: pd.Series):
    """Compute cuae-metric.

    Args:
        y_true - stands for the true label
        y_pred - predictions
        sensitive_features - sensitive attribute
    Returns:
        dictionary with cuae-metric df and cuae-difference.
    """
    df = pd.DataFrame({'true': np.array(y_true), 'pred': np.array(y_pred),
                       'protected': np.array(sensitive_features)}).astype('category')
    classes = df['true'].drop_duplicates()
    protected_groups_values = df['protected'].drop_duplicates()
    np_ans = np.zeros(shape=[len(protected_groups_values), len(classes)])
    for j in range(len(protected_groups_values)):
        for i in range(len(classes)):
            protected_value = protected_groups_values[protected_groups_values.index[j]]
            current_part = df[df['protected'] == protected_value]
            ndf = current_part[(current_part['true'] == classes[classes.index[i]])]
            res = accuracy_score(ndf['true'], ndf['pred'])
            np_ans[j, i] = res
    df = pd.DataFrame(np_ans, columns=np.array(classes), index=np.array(protected_groups_values))

    return {'df': df, 'diff': max([df[x].max() - df[x].min() for x in df.columns])}
