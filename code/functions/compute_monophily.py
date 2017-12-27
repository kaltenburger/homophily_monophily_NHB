# reviewed: 11/14/2017
# 4/24/2017
# KM Altenburger
# monophily_index: computes class-specific monophily measure
# baseline_monophily_index: computes baseline class-specific monophily measure


from __future__ import division
import rpy2.robjects
import rpy2.robjects.numpy2ri
import rpy2.robjects as ro
rpy2.robjects.numpy2ri.activate()


rpy2.robjects.r('''
    library(dispmod)
    f_exact_p_value_intercept <- function(mod){
    mod <- glm(cbind(deg_same, deg_different) ~ 1, family=binomial(logit))
    mod.disp <- glm.binomial.disp(mod, maxit = 50, verbose = F)
    mod.disp$dispersion
    }
    ''')

rpy2.robjects.r('''
    library(dispmod)
    f <- function(deg_same, deg_different){
        mod <- glm(cbind(deg_same, deg_different) ~ 1, family=binomial(logit))
        mod.disp <- glm.binomial.disp(mod, maxit = 50, verbose = F)
        mod.disp$dispersion
    }
    ''')



rpy2.robjects.r('''
    library(dispmod)
    f_intercept <- function(deg_same, deg_different){
        mod <- glm(cbind(deg_same, deg_different) ~ 1, family=binomial(logit))
        mod.disp <- glm.binomial.disp(mod, maxit = 50, verbose = F)
        return(list(as.numeric(coef(mod)[1]),as.numeric(coef(mod.disp))[1]))
    }
    ''')



## compute standard errors
rpy2.robjects.r('''
    library(MASS)
    f_p_value_99 <- function(deg_same, deg_different){
    mod <- glm(cbind(deg_same, deg_different) ~ 1, family=binomial(logit))
    return(confint(mod, level = 0.99))
    }
    ''')

rpy2.robjects.r('''
    library(MASS)
    f_p_value_99_9 <- function(deg_same, deg_different){
    mod <- glm(cbind(deg_same, deg_different) ~ 1, family=binomial(logit))
    return(confint(mod, level = 0.999))
    }
    ''')


r_f = rpy2.robjects.r['f']
r_intercept = rpy2.robjects.r['f_intercept']
r_standard_errors_99 = rpy2.robjects.r['f_p_value_99']
r_standard_errors_99_9 = rpy2.robjects.r['f_p_value_99_9']

def monophily_index_overdispersion_Williams_with_intercept_SE_99(adj_matrix, membership_vector):
    num_labels = len(np.unique(np.array(membership_vector)))
    class_labels = np.sort(np.unique(np.array(membership_vector)))
    monophily_intercept_SE_by_class = []
    for j in range(num_labels):
        ## among users of class label 'j' -- find # of their friends also of class label 'j'
        same_j = adj_matrix[np.array(membership_vector)==class_labels[j],] * np.transpose(np.matrix(np.array(membership_vector)==class_labels[j])+0)
        
        ## among users of class label 'j' -- find # of their friends NOT of class label 'j'
        different_j = adj_matrix[np.array(membership_vector)==class_labels[j],] * np.transpose(np.matrix(np.array(membership_vector)!=class_labels[j])+0)
        
        r_same = ro.r.array(np.array(same_j).T[0])
        ro.r.assign("r_same", r_same)
        r_diff = ro.r.array(np.array(different_j).T[0])
        ro.r.assign("r_diff", r_diff)
        monophily_intercept_SE_by_class.append( np.array(r_standard_errors_99(r_same, r_diff)))
    return(np.array(monophily_intercept_SE_by_class).T)


def monophily_index_overdispersion_Williams_with_intercept_SE_99_9(adj_matrix, membership_vector):
    num_labels = len(np.unique(np.array(membership_vector)))
    class_labels = np.sort(np.unique(np.array(membership_vector)))
    monophily_intercept_SE_by_class = []
    for j in range(num_labels):
        ## among users of class label 'j' -- find # of their friends also of class label 'j'
        same_j = adj_matrix[np.array(membership_vector)==class_labels[j],] * np.transpose(np.matrix(np.array(membership_vector)==class_labels[j])+0)
        
        ## among users of class label 'j' -- find # of their friends NOT of class label 'j'
        different_j = adj_matrix[np.array(membership_vector)==class_labels[j],] * np.transpose(np.matrix(np.array(membership_vector)!=class_labels[j])+0)
        r_same = ro.r.array(np.array(same_j).T[0])
        ro.r.assign("r_same", r_same)
        r_diff = ro.r.array(np.array(different_j).T[0])
        ro.r.assign("r_diff", r_diff)
        monophily_intercept_SE_by_class.append( np.array(r_standard_errors_99_9(r_same, r_diff)))
    return(np.array(monophily_intercept_SE_by_class).T)



def monophily_index_overdispersion_Williams_with_intercept(adj_matrix, membership_vector):
    num_labels = len(np.unique(np.array(membership_vector)))
    class_labels = np.sort(np.unique(np.array(membership_vector)))
    monophily_intercept_by_class = []
    for j in range(num_labels):
        ## among users of class label 'j' -- find # of their friends also of class label 'j'
        same_j = adj_matrix[np.array(membership_vector)==class_labels[j],] * np.transpose(np.matrix(np.array(membership_vector)==class_labels[j])+0)
        
        ## among users of class label 'j' -- find # of their friends NOT of class label 'j'
        different_j = adj_matrix[np.array(membership_vector)==class_labels[j],] * np.transpose(np.matrix(np.array(membership_vector)!=class_labels[j])+0)
        r_same = ro.r.array(np.array(same_j).T[0])#, nrow=nr, ncol=nc)
        ro.r.assign("r_same", r_same)
        r_diff = ro.r.array(np.array(different_j).T[0])#, nrow=nr, ncol=nc)
        ro.r.assign("r_diff", r_diff)
        monophily_intercept_by_class.append( np.array(r_intercept(r_same, r_diff)))
    return(np.array(monophily_intercept_by_class).T[0])



def monophily_index_overdispersion_Williams(adj_matrix, membership_vector):
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
        monophily_index_by_class.append( np.float(np.array(r_f(r_same, r_diff))[0]))
    return(monophily_index_by_class)
