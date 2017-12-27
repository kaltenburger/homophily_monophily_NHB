# code adapted from Sam F. Way

import itertools
from numpy import zeros
from scipy.sparse import coo_matrix


def compute_mixing_matrix(A, metadata_values):
    """ Compute the mixing matrix for the provided 
        adjacency matrix and labels """ 

    if A.shape[0] != len(metadata_values):
        raise ValueError('Adjacency matrix and metadata vector must'
                         ' have the same dimension!')

    unique_values = sorted(list(set(metadata_values)))
    num_unique = len(unique_values)
    value_map = {v:i for i,v in enumerate(unique_values)}
    value_key = {i:v for i,v in enumerate(unique_values)}
    values = [value_map[x] for x in metadata_values]

    M = zeros((num_unique, num_unique))

    if hasattr(A, 'tocoo'):
        edges = A.tocoo()
    else:
        edges = coo_matrix(A)

    for i,j,v in itertools.izip(edges.row, edges.col, edges.data):
        s = values[i]
        d = values[j]
        M[s,d] += 1.0
        # i,j and j,i are both in the edgelist, so don't worry about M[d,s]

    M /= M.sum()
    return M, value_key

