from __future__ import division

## 10/16/2017
## about: function to create DIRECTED adjacency matrix -- with option of whether to create diagonal; pre-processing: If >1 connected component - we work with largest connected one.


## Returns: nxn symmetric matrix -- since ZGL requires symmetric matrices
## create function given graph and dictionary -- return adj_matrix and dictionary
def create_directed_adj_membership(graph, dictionary,  val_to_drop, delete_na_cols, diagonal, directed_type, attribute):
    keys = np.array(range(len(dictionary.keys()))) ## update keys
    y_vector = np.array(dictionary.values()) ## we relabel keys but this preserves corresponding value with updated keys
    adj_matrix_input = nx.adj_matrix(graph,
                                     nodelist = dictionary.keys()).todense() # note: will automatically be an out-link matrix when graph is directed
    nx.set_node_attributes(graph,attribute,dictionary)
    
    ## spot-check: confirm nxn matrix
    if adj_matrix_input.shape[0]!=adj_matrix_input.shape[1]:
        print 'error: must be nxn matirx'
        return

    
    ## create in-link matrix
    ## y_vector stays same; keys stay same
    if directed_type == 'in':
        adj_matrix_input = nx.adj_matrix(graph,
                                         nodelist = dictionary.keys()).todense().T


    if diagonal == 1:
        #print adj_matrix_input.shape
        adj_matrix_input[range(adj_matrix_input.shape[0]),range(adj_matrix_input.shape[0])] = 1

    ## remove NA labeled nodes
    if np.sum((y_vector==val_to_drop)+0) > 0: # first confirm there are >0 NA labeled nodes
        keys = keys[y_vector!=val_to_drop]
        adj_matrix_input = adj_matrix_input[np.array(keys),:]
        y_vector = y_vector[y_vector!=val_to_drop]
    
        if delete_na_cols == 'yes': ## and remove NA nodes in column too
            adj_matrix_input=adj_matrix_input[:,np.array(keys)]


    A_final = np.copy(adj_matrix_input)
    y_vector_full_columns = np.copy(y_vector)

    if np.sum((np.sum(A_final,1)==0)+0)>0:
        degree_0 = 1
        while degree_0 > 0:
            subset_training_test_deg_0 = np.array(range((A_final.shape[0])))[np.array(np.sum(A_final,1)!=0).ravel()]
            A_cv = A_final[np.array(subset_training_test_deg_0),:] # remove as row
            A_cv = A_cv[:,np.array(subset_training_test_deg_0)]
            y_vector = y_vector[np.array(subset_training_test_deg_0)]
            A_final = np.copy(A_cv) # only drop NA nodes as rows and NOT as columns
            degree_0= np.sum((np.sum(A_final,1)==0)+0)


    return(np.array(y_vector), np.matrix(A_final))


