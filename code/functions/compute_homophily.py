from __future__ import division

# 4/24/2017
# about: compute homophily index

def homophily_index_Jackson_alternative(adj_matrix, membership_vector):
    ##http://web.stanford.edu/~jacksonm/netminority.pdf
    # get number of unique labels in membership vector
    # outputs observed homophily where values are ordered by class
    num_labels = len(np.unique(np.array(membership_vector)))
    class_labels = np.sort(np.unique(np.array(membership_vector)))
    homophily_index_by_class = []
    for j in range(num_labels):
        ## among users of class label 'j' -- find # of their friends also of class label 'j'
        same_j = adj_matrix[np.array(membership_vector)==class_labels[j],] * np.transpose(np.matrix(np.array(membership_vector)==class_labels[j])+0)
        
        ## among users of class label 'j' -- find # of their friends NOT of class label 'j'
        different_j = adj_matrix[np.array(membership_vector)==class_labels[j],] * np.transpose(np.matrix(np.array(membership_vector)!=class_labels[j])+0)
        homophily_index_by_class.append( np.mean(same_j)/np.mean(same_j+different_j))
    return(homophily_index_by_class)
