from __future__ import division
import rpy2.robjects
import rpy2.robjects.numpy2ri
import rpy2.robjects as ro
rpy2.robjects.numpy2ri.activate()



rpy2.robjects.r('''
    library(dispmod)
    f_pval <- function(deg_same, deg_different){
    #print(cbind(deg_same, deg_different, rowSums(data.frame(deg_same, deg_different))))
    mod <- glm(cbind(deg_same, deg_different) ~ 1, family=binomial(logit))
    #print(summary(mod))
    return(coef(summary(mod))[,4])
    }
    ''')
r_f_pvalue = rpy2.robjects.r['f_pval']



rpy2.robjects.r('''
    library(dispmod)
    f_p <- function(deg_same, deg_different){
    mod <- glm(cbind(deg_same, deg_different) ~ 1, family=binomial(logit))
    return(coef(summary(mod))[,1])
    
    }
    ''')
r_f_p = rpy2.robjects.r['f_p']

def homophily_intercept_p_value(adj_matrix, membership_vector):
    num_labels = len(np.unique(np.array(membership_vector)))
    class_labels = np.sort(np.unique(np.array(membership_vector)))
    homophily = []
    for j in range(num_labels):
        ## among users of class label 'j' -- find # of their friends also of class label 'j'
        same_j = adj_matrix[np.array(membership_vector)==class_labels[j],] * np.transpose(np.matrix(np.array(membership_vector)==class_labels[j])+0)
        
        ## among users of class label 'j' -- find # of their friends NOT of class label 'j'
        different_j = adj_matrix[np.array(membership_vector)==class_labels[j],] * np.transpose(np.matrix(np.array(membership_vector)!=class_labels[j])+0)
        
        r_same = ro.r.array(np.array(same_j).T[0])
        ro.r.assign("r_same", r_same)
        r_diff = ro.r.array(np.array(different_j).T[0])
        ro.r.assign("r_diff", r_diff)
        homophily.append( np.array(r_f_pvalue(r_same, r_diff)))
    return(np.array(homophily))

def homophily_intercept(adj_matrix, membership_vector):
    num_labels = len(np.unique(np.array(membership_vector)))
    class_labels = np.sort(np.unique(np.array(membership_vector)))
    homophily = []
    for j in range(num_labels):
        ## among users of class label 'j' -- find # of their friends also of class label 'j'
        same_j = adj_matrix[np.array(membership_vector)==class_labels[j],] * np.transpose(np.matrix(np.array(membership_vector)==class_labels[j])+0)
        
        ## among users of class label 'j' -- find # of their friends NOT of class label 'j'
        different_j = adj_matrix[np.array(membership_vector)==class_labels[j],] * np.transpose(np.matrix(np.array(membership_vector)!=class_labels[j])+0)
        
        r_same = ro.r.array(np.array(same_j).T[0])
        ro.r.assign("r_same", r_same)
        r_diff = ro.r.array(np.array(different_j).T[0])
        ro.r.assign("r_diff", r_diff)
        homophily.append( np.array(r_f_p(r_same, r_diff)))
    return(np.array(homophily))

