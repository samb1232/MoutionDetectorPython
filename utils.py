import numpy as np


def remove_nan(arr):
    # Detect nan values
    nan_indices = np.isnan(arr).any(axis=1)

    # Remove rows with nan values
    return arr[~nan_indices]

