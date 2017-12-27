# code adapted from Sam F. Way

from scipy.io import loadmat
from numpy import genfromtxt


def parse_fb100_mat_file(filename):
    """ Parse FB100 .mat files.

        Metadata values:
            0 - student/faculty status
            1 - gender
            2 - major
            3 - second major/minor
            4 - dorm/house
            5 - year
            6 - high school
        ** Missing data coded as 0 **
        
        Parameters:
        - filename = path to FB100 .mat file

        Returns:
        (adj_matrix, metadata)
        - adj_matrix = adjacency matrix for the network
        - metadata = matrix containing metadata for each node

    """
    mat = loadmat(filename)
    error_msg = "%s is not a valid FB100 .mat file.  Must " \
                "contain data for variable '%s'"
    
    if 'A' not in mat:
        raise ValueError(error_msg % (filename, 'A'))
    adj_matrix = mat['A']
    
    if 'local_info' not in mat:
        raise ValueError(error_msg % (filename, 'local_info'))
    metadata = mat['local_info']

    return adj_matrix, metadata


def parse_mixing_matrix_file(filename):
    """ Parse mixing matrix made with make_mixing_matrices.py
        
        Returns a numpy matrix and a key describing the original
        metadata value associated with each row/column index.
    """
    handle = open(filename, 'rU')
    header = handle.readline().strip()
    if header[0] != '#':
        raise ValueError('%s has an unexpected format' % filename)
    header = header[1:]
    keys = [int(x) for x in header.split(',')]
    handle.close()
    
    M = genfromtxt(filename, delimiter=',')
    
    return M, keys
