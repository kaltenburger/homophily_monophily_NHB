# created: 8/21/2017
# edited: 11/6/2017 for final inclusion in NHB revision
# KM Altenburger

# About:
# monophily_index: computes class-specific monophily measure (assuming latent beta distribution on p's)
# baseline_monophily_index: computes baseline class-specific monophily measure
# note that the model is fit for each class to be consistent with the class-specific homophily index

from __future__ import division
import rpy2.robjects
import rpy2.robjects.numpy2ri
import rpy2.robjects as ro
rpy2.robjects.numpy2ri.activate()

rpy2.robjects.r('''
    library(aod)
    beta_bin <- function(deg_same, deg_different){
            tmp <- data.frame(deg_same, deg_different)
            model_bb <- betabin(cbind(deg_same, deg_different) ~ 1,
                        ~1,
                        warnings = FALSE,
                        data = tmp)
            as.vector(model_bb@param[2])
    }
    ''')
r_bb = rpy2.robjects.r['beta_bin']


rpy2.robjects.r('''
    library(aod)
    beta_bin_homophily <- function(deg_same, deg_different){
    tmp <- data.frame(deg_same, deg_different)
    model_bb <- betabin(cbind(deg_same, deg_different) ~ 1,
    ~1,
    warnings = FALSE,
    data = tmp)
    as.vector(model_bb@param[1])
    }
    ''')
r_bb_homophily = rpy2.robjects.r['beta_bin_homophily']


def monophily_index_beta_bin(adj_matrix, membership_vector):
    num_labels = len(np.unique(np.array(membership_vector)))
    class_labels = np.sort(np.unique(np.array(membership_vector)))
    monophily_index_by_class = []
    for j in range(num_labels):
        ## among users of class label 'j' -- find # of their friends also of class label 'j'
        same_j = adj_matrix[np.array(membership_vector)==class_labels[j],] * np.transpose(np.matrix(np.array(membership_vector)==class_labels[j])+0)
        
        ## among users of class label 'j' -- find # of their friends NOT of class label 'j'
        different_j = adj_matrix[np.array(membership_vector)==class_labels[j],] * np.transpose(np.matrix(np.array(membership_vector)!=class_labels[j])+0)
        r_same = ro.r.array(np.array(same_j).T[0])
        ro.r.assign("r_same", r_same)
        r_diff = ro.r.array(np.array(different_j).T[0])
        ro.r.assign("r_diff", r_diff)
        monophily_index_by_class.append( np.float(np.array(r_bb(r_same, r_diff))[0]))
    return(monophily_index_by_class)



def homophily_index_beta_bin(adj_matrix, membership_vector):
    num_labels = len(np.unique(np.array(membership_vector)))
    class_labels = np.sort(np.unique(np.array(membership_vector)))
    homophily_index_by_class = []
    for j in range(num_labels):
        ## among users of class label 'j' -- find # of their friends also of class label 'j'
        same_j = adj_matrix[np.array(membership_vector)==class_labels[j],] * np.transpose(np.matrix(np.array(membership_vector)==class_labels[j])+0)
        
        ## among users of class label 'j' -- find # of their friends NOT of class label 'j'
        different_j = adj_matrix[np.array(membership_vector)==class_labels[j],] * np.transpose(np.matrix(np.array(membership_vector)!=class_labels[j])+0)
        r_same = ro.r.array(np.array(same_j).T[0])
        ro.r.assign("r_same", r_same)
        r_diff = ro.r.array(np.array(different_j).T[0])
        ro.r.assign("r_diff", r_diff)
        homophily_index_by_class.append( np.float(np.array(r_bb_homophily(r_same, r_diff))[0]))
    return(homophily_index_by_class)
