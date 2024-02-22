import numpy as np
import pandas as pd


def _zeros_ones_to_classes(x: np.array, length: int = 3):
    """Auxiliary function for core._lp_solver. Builds array to make fair predictions given solution of lp problem."""
    n = int(len(x) / length)

    return np.array([x[i * length: i * length + length].argmax() for i in range(n)], dtype=int)


def _answer_creator(x: list, y: list, grouper: pd.Series):
    """Auxiliary function for core._predictor. Make prediction given prediction for 0 and 1 classes and grouper."""
    x = np.array(x)  # array of 1
    y = np.array(y)  # array of 0
    grouper = np.array(grouper)
    ans = []
    x_ind = 0
    y_ind = 0

    for i in range(len(grouper)):
        if grouper[i] == 0:
            ans.append(y[y_ind])
            y_ind += 1
        else:
            ans.append(x[x_ind])
            x_ind += 1

    return np.array(ans)
