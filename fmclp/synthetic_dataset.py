import numpy as np
import pandas as pd


def synthetic_dataset(size: int = 1000, influence: bool = True):
    """Create synthetic dataset with or without dependence between target and sensitive attribute.

    Args:
        size - number of observations
        influence - presence of dependence between sensitive attribute and label.
    Returns:
          pd.DataFrame with synthetic data.
    """

    attr = np.random.choice([0, 1], size=size)
    error_x = np.random.normal(loc=0.0, scale=0.3, size=size)
    error_y = np.random.normal(loc=0.0, scale=0.3, size=size)
    error_z = np.random.normal(loc=0.0, scale=0.3, size=size)
    error_target = np.random.normal(loc=0.0, scale=0.5, size=size)

    y1 = np.random.normal(loc=1, scale=1, size=size)
    y2 = np.random.normal(loc=1, scale=1, size=size)
    y3 = np.random.normal(loc=1, scale=1, size=size)

    x = y1 + y2 + error_x
    y = y1 + y3 + error_y
    z = y2 + y3 + error_z

    if influence:
        target = x * (1 + 2 * attr) + y * (1 - 0.5 * attr) + z * (1 + 0.5 * attr) + error_target * attr
    else:
        target = x + y + z + error_target

    target = _simple_splitter(target)

    synthetic_df = pd.DataFrame(np.array((x, y, z, attr, target))).T.rename(
        columns={0: 'x', 1: 'y', 2: 'z', 3: 'attr', 4: 'target'})

    return synthetic_df


def _simple_splitter(arr: np.array):
    """Auxiliary function to build synthetic dataset."""
    arr_unchanged = arr.copy()
    arr = np.sort(np.array(arr))
    length = len(arr)
    n1 = arr[int(length / 3)]
    n2 = arr[int(2 * length / 3)]
    result = []

    for i in range(length):
        if arr_unchanged[i] <= n1:
            result.append(0)
        elif n1 < arr_unchanged[i] <= n2:
            result.append(1)
        else:
            result.append(2)

    return np.array(result)
