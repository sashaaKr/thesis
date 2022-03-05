import numpy as np

def get_n_indexes_of_max_values(arr, n):
    indices = np.argpartition(-arr, n)[:n]
    sorted_indices = indices[np.argsort(arr[indices])]
    return sorted_indices

def get_max_similarity_per_p(similarities):
    res = []
    for index, value in enumerate(similarities):
        max_indices = get_n_indexes_of_max_values(value, 6)
        max_indices_without_self = max_indices[:-1]
        max_similarity = value[max_indices_without_self[-1]]
        res.append(max_similarity)
    return res