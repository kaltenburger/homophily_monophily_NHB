## 4/24/2017
## about: compute null distribution given input adjacency matrix and corresponding labels for the full network

from __future__ import division
from scipy import stats

def compute_observed_same_total_degree(adj_matrix, membership_vector):
    num_labels = len(np.unique(np.array(membership_vector)))
    class_labels = np.sort(np.unique(np.array(membership_vector)))
    
    observed_deg_by_class = []
    for j in range(num_labels):
        total_degree_for_class = np.sum(adj_matrix[membership_vector==class_labels[j],] ,1)
        d_i = map(np.int,np.array(total_degree_for_class.T)[0])
        in_degree = adj_matrix[membership_vector==class_labels[j],] * np.matrix((membership_vector==class_labels[j])+0).T
        observed_deg_by_class.append(in_degree/total_degree_for_class)
    return(observed_deg_by_class)


def compute_null_distribution(adj_matrix, membership_vector, n_iter):
    num_labels = len(np.unique(np.array(membership_vector)))
    class_labels = np.sort(np.unique(np.array(membership_vector)))
    
    preference_by_class = []
    for j in range(num_labels):
        total_degree_for_class = np.sum(adj_matrix[membership_vector==class_labels[j],] ,1)
        d_i = map(np.int,np.array(total_degree_for_class.T)[0])
        in_degree = adj_matrix[membership_vector==class_labels[j],] * np.matrix((membership_vector==class_labels[j])+0).T

        #compute homophily index
        h_index = np.mean(in_degree)/np.mean(total_degree_for_class)

        mc_distribution_tmp = []
        for j in range(n_iter):
            mc_distribution_tmp.append( np.random.binomial(n=map(np.int,np.array(total_degree_for_class.T)[0]),
                                                       p=h_index)/d_i)
        
        mc_distribution = np.array(mc_distribution_tmp).flatten()
        preference_by_class.append(mc_distribution)
        
    return(preference_by_class)
