import numpy as np

def get_n_indexes_of_max_values(arr, n):
    indices = np.argpartition(-arr, n)[:n]
    sorted_indices = indices[np.argsort(arr[indices])]
    return sorted_indices