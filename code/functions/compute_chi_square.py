# 12/27/2016
# KM Altenburger
# about: compute chi-square goodness of fit statistic

from __future__ import division
from scipy import stats


## Crowder/78 in 2nd paragraph discusses chi-square
## chi-square statistic ==  sum{ [d_iF - d_i * H_f]**2 / (d_i * H_f * (1-H_F)) }
## degrees of freedom (assuming an intercept-only model) is n_F -1 because we use 1 df to estimate intercept term
def compute_chi_square_statistic(adj_matrix, membership_vector):
    num_labels = len(np.unique(np.array(membership_vector)))
    class_labels = np.sort(np.unique(np.array(membership_vector)))
    chi_square_index_by_class = []
    for j in range(num_labels):
        ## among users of class label 'j' -- find # of their friends also of class label 'j'
        same_j = adj_matrix[np.array(membership_vector)==class_labels[j],] * np.transpose(np.matrix(np.array(membership_vector)==class_labels[j])+0)
        ## among users of class label 'j' -- find # of their friends NOT of class label 'j'
        different_j = adj_matrix[np.array(membership_vector)==class_labels[j],] * np.transpose(np.matrix(np.array(membership_vector)!=class_labels[j])+0)
        total_degree = same_j+different_j
        H_class = np.mean(same_j)/np.mean(same_j+different_j)
        temp1 = np.array((same_j - H_class * total_degree)).T[0]
        temp2 = np.array(total_degree * H_class * (1-H_class)).T[0]
        df = np.sum((np.array(membership_vector)==class_labels[j])+0) - 1 # df = n_r - 1
        chi_square_index_by_class.append(1-stats.chi2.cdf(np.sum(temp1**2/(temp2)), df))
    return(chi_square_index_by_class)
