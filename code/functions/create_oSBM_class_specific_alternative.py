from __future__ import division

## 12/18/2016
## update: allow class-specific dispersion parameter
## note: assumes dispersion_val_vect is in the same order as the class [class1, class2] --> dispersion_val_vect = [dispersion_val_vect_1, dispersion_val_vect_2]
## simply put, the numeric ordering of the class determines the ordering of the


def create_proportion_class_k_friends(adj_matrix, node_id,
                                     y_labels, k_class):
    prop_class_k_friends = []
    total_neighbors = []
    total = np.sum(adj_matrix,1)
    class_k_num = adj_matrix*np.matrix((y_labels==k_class)+0).T
    prop_class_k =class_k_num/total
    return(np.array(prop_class_k).T[0])





def create_expected_degree_sequence(class_size_val,
                                    p_in_val, p_out_val,
                                    dispersion_val_in, dispersion_val_out):
    ## in-class
    if(dispersion_val_in != 0):
        alpha_in = p_in_val * (1/dispersion_val_in) * (1-dispersion_val_in)
        beta_in = (1-p_in_val) * (1/dispersion_val_in) * (1-dispersion_val_in)
        p_in_dispersed = np.matrix(np.random.beta(alpha_in, beta_in, size=class_size_val))
        in_class_expected_degree = p_in_dispersed * class_size_val # probability of link * number of possible in-LINKS
    if(dispersion_val_in == 0):
        in_class_expected_degree = np.matrix([class_size_val * p_in_val] * class_size_val)

    ## out-class
    if(dispersion_val_out != 0):
        alpha_out = p_out_val * (1/dispersion_val_out) * (1-dispersion_val_out)
        beta_out = (1-p_out_val) * (1/dispersion_val_out) * (1-dispersion_val_out)
        p_out_dispersed = np.matrix(np.random.beta(alpha_out, beta_out, size=class_size_val))
        out_class_expected_degree = p_out_dispersed * class_size_val # probability of link * number of possible out-LINKS
    if(dispersion_val_out == 0):
        out_class_expected_degree = np.matrix([class_size_val * p_out_val] * class_size_val)
    return(in_class_expected_degree, out_class_expected_degree)



def in_class_matrix(matrix):
    return(matrix.T*matrix)

def out_class_matrix(matrix1,matrix2):
    return(matrix1.T*matrix2)



## assumes k=2 class set-up
## p_in = [p_in_1, p_in_2]
def create_affiliation_model_temp(average_node_degree,
                                  lambda_block_parameter,
                                  dispersion_parameter_vect,
                                  class_size_vect):
    N = np.sum(class_size_vect)
    ### BLOCK STRUCTURE
    ## define p_in; p_out
    p_in = (lambda_block_parameter * average_node_degree)/N
    print 'p_in: ', p_in
    #previous parameterization
    denominator = []
    for j in range(len(class_size_vect)):
        denominator.append(class_size_vect[j] * class_size_vect[~j])
    denom = np.sum(denominator)
    p_out = (average_node_degree * N - np.sum(class_size_vect**2 * p_in))/denom
    print 'p_out: ', p_out
    print ''

    ## Expected Degree Sequence for nodes in class 1,2,...k
    ## Generates in-class degree sequence and out-class sequence
    in_class_list = []
    out_class_list = []
    for j in range(len(class_size_vect)):
        #intent here is to iterate through each class
        #and important -- assumes a specific data format for input dispersion_parameter_vect
        (in_class, out_class) = create_expected_degree_sequence(class_size_vect[j],p_in,p_out,dispersion_parameter_vect[j][0], dispersion_parameter_vect[j][1])
        in_class_list.append(in_class)
        out_class_list.append(out_class)


    expected_prob_matrix=np.zeros((N,N))
    for i in range(len(class_size_vect)):
        for j in range(len(class_size_vect)):
            idx = np.sum(class_size_vect[0:i])
            jdx = np.sum(class_size_vect[0:j])
            if i==j:
                expected_prob_matrix[idx:idx+class_size_vect[j],jdx:jdx+class_size_vect[j]] = in_class_matrix(in_class_list[j])/(class_size_vect[j]**2*p_in)
            else:
                out = out_class_matrix(out_class_list[i], out_class_list[j])/(class_size_vect[i]*class_size_vect[j]*p_out)
                if j<i:
                    expected_prob_matrix[idx:idx+class_size_vect[i],jdx:jdx+class_size_vect[j]] = out
                if i<j:
                    expected_prob_matrix[idx:idx+class_size_vect[i],jdx:jdx+class_size_vect[j]] = out
    A_ij_tmp = np.matrix(map(bernoulli.rvs,expected_prob_matrix))
    Adj_corrected = np.matrix(np.triu(A_ij_tmp, k=0) + np.transpose(np.triu(A_ij_tmp, k=1)))
    Membership = np.concatenate(map(np.tile,np.array(range(len(class_size_vect))), class_size_vect),0)

    print 'spot-check average degree: '
    print np.mean(np.sum(np.matrix(Adj_corrected), axis=1))
    print ''

    print 'spot-check homophily: '
    print homophily_index_Jackson_alternative(np.matrix(Adj_corrected), np.array(Membership))
    print ''

    print 'spot-check monophily: '
    print monophily_index_overdispersion_Williams(np.matrix(Adj_corrected), np.array(Membership))
    print ''
    return( Adj_corrected, Membership)
