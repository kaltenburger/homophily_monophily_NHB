from __future__ import division

## edited: 11/8/2017
## 4/24/2017
## about: function to create adjacency matrix for UNDIRECTED networks -- with option of whether to create self-loops by setting diagonal==1; pre-processing: if there is >1 connected component -- we work with largest connected one.
## returns: nxn symmetric matrix -- since ZGL requires symmetric matrices; and nx1 vector of attributes


## workflow:
##  1) remove nodes with NA labels
##  2) subset to largest connected component
##  3) drop nodes with degree == 0 [spot-check].

def create_adj_membership(graph, dictionary,  val_to_drop, delete_na_cols, diagonal, directed_type, attribute):
    keys = np.array(range(len(dictionary.keys())))      ## update keys
    y_vector = np.array(dictionary.values())            ## we relabel keys but this preserves corresponding value with updated keys
    adj_matrix_input = nx.adj_matrix(graph,
                                     nodelist = dictionary.keys()).todense()   # note: will automatically be an out-link matrix when graph is directed
    nx.set_node_attributes(graph,attribute,dictionary)
    
    ## spot-check: confirm nxn matrix
    if adj_matrix_input.shape[0]!=adj_matrix_input.shape[1]:
        print 'error: must be nxn matirx'
        return
    
    
    ## function setting to permit self-loops
    if diagonal == 1:
        adj_matrix_input[range(adj_matrix_input.shape[0]),range(adj_matrix_input.shape[0])] = 1
    
    ## remove NA labeled nodes
    if np.sum((y_vector==val_to_drop)+0) > 0: # first confirm there are >0 NA labeled nodes
        keys = keys[y_vector!=val_to_drop]
        adj_matrix_input = adj_matrix_input[np.array(keys),:]
        y_vector = y_vector[y_vector!=val_to_drop]
        if delete_na_cols == 'yes': ## and remove NA nodes in column too
            adj_matrix_input=adj_matrix_input[:,np.array(keys)]

        #update
        graph = nx.from_numpy_matrix(adj_matrix_input)
        attr_new = create_dict(range(adj_matrix_input.shape[0]),
                               y_vector)
        nx.set_node_attributes(graph,attribute,attr_new)



    ## create undirected network
    ## subset to nodes only in largest connected component
    if directed_type == None:
        if nx.number_connected_components(graph) > 1:
            max_cc = np.max(map(len,map(nx.nodes,nx.connected_component_subgraphs(graph, copy=True))))
            subset_cc =  map(len,map(nx.nodes,nx.connected_component_subgraphs(graph, copy=True)))==max_cc
            largest_cc_subgraph = np.array(list(nx.connected_component_subgraphs(graph, copy=True)))[subset_cc][0]
            adj_matrix_input = nx.adj_matrix(largest_cc_subgraph,
                                             nodelist = nx.get_node_attributes(largest_cc_subgraph, attribute).keys()).todense()
            y_vector = np.array(nx.get_node_attributes(largest_cc_subgraph, attribute).values())
            keys = np.array(range(len(np.array(nx.get_node_attributes(largest_cc_subgraph, attribute).keys())))) ## update keys

    A_final = np.copy(adj_matrix_input)

    ## undirected: remove degree == 0 nodes
    if np.sum((np.sum(A_final,1)==0)+0)>0 and directed_type != 'in' and directed_type != 'out':
        subset_training_test_deg_0 = np.array(range((A_final.shape[0])))[np.array(np.sum(A_final,1)!=0).ravel()]
        A_cv = A_final[np.array(subset_training_test_deg_0),:] # remove as row
        A_cv = A_cv[:,np.array(subset_training_test_deg_0)] # remove as column
        y_vector = y_vector[np.array(subset_training_test_deg_0)]
        A_final = np.copy(A_cv) # only drop NA nodes as rows and NOT as columns

    return(np.array(y_vector), np.matrix(A_final))


