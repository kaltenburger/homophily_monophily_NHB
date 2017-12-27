## 11/30/2017
## KM Altenburger

from __future__ import division
import os

## compare AUC values @k-hop for fully-labeled network, examining _all_ AH schools

## how to run:
## cd /Users/kristen/Dropbox/gender_graph_data/manuscript/nature_hb/gender_graph_final_code_NatureHB/code/functions
## python compare_k_hop_friends_vs_AUC_add_health.py -i='/Users/kristen/Dropbox/gender_graph_data/add-health/converted_gml' -o='/Users/kristen/Dropbox/gender_graph_data/manuscript/nature_hb/gender_graph_final_code_NatureHB/data/output'
import os
folder_directory =os.getcwd()
print(folder_directory)
os.chdir(folder_directory)

execfile('python_libraries.py')
execfile('create_adjacency_matrix.py')
execfile('parsing.py')  # Sam Way's Code
execfile('mixing.py')   # Sam Way's Code

def interface():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input-dir', help='Input directory', required=True)
    args.add_argument('-e', '--file-ext', help='Input extension', default='.gml')
    args.add_argument('-o', '--output-dir', help='Output directory', required=True)
    args = args.parse_args()
    return args

if __name__=="__main__":
    args = interface()
    #print args
    homophily_gender = []
    monophily_gender = []
    
    
    file_output = open('../../data/output/khop_vs_auc_add_health_undirected_Nov2017.csv', 'wt')
    j =0
    writer = csv.writer(file_output)
    writer.writerow( ('school', 'k_hop', 'auc_in_khop_neighborhood', 'count_0_friends_khop_class1','count_0_friends_khop_class2'))
                      
    os.chdir('/Users/kristen/Dropbox/gender_graph_data/add-health/converted_gml/')
    for f in listdir(args.input_dir):
        if f.endswith(args.file_ext):
            tag = f.replace(args.file_ext, '')
            print tag
            j=j+1
            if (tag!='schools'):
                print "Processing %s..." % tag
                
                tag = f.replace(args.file_ext, '')
                id = re.findall(r'\d+', f)
            
                # undirected graph
                ah_graph_tmp = nx.read_gml(f)
                ah_graph = ah_graph_tmp.to_undirected()
            
                (gender_y, adj_matrix_gender) = create_adj_membership(ah_graph,
                                                                                           nx.get_node_attributes(ah_graph, 'comm' + str(id[0]) +'sex'),
                                                                                           0,
                                                                                           'yes',
                                                                                           0,
                                                                                           None,
                                                                                    'gender')
                k_hop = np.array([1,2,3,4,5])
                class_values = np.sort(np.unique(gender_y))
                

                for k in k_hop:
                    adj_amherst_k= np.matrix(adj_matrix_gender)**k
                    adj_amherst_k[range(adj_amherst_k.shape[0]),range(adj_amherst_k.shape[0])]=0 ## remove self-loops
                    
                    nonzero_idx1 = np.array((np.sum(adj_amherst_k[gender_y==class_values[0],:],1)!=0).T)[0]
                    nonzero_idx2 = np.array((np.sum(adj_amherst_k[gender_y==class_values[1],:],1)!=0).T)[0]
                    
                    mv_g1 = (adj_amherst_k[gender_y==class_values[0],:] * np.matrix((gender_y==class_values[0])+0).T)[nonzero_idx1]/np.sum(adj_amherst_k[gender_y==class_values[0],:],1)[nonzero_idx1]
                    mv_g2 = (adj_amherst_k[gender_y==class_values[1],:] * np.matrix((gender_y==class_values[1])+0).T)[nonzero_idx2]/np.sum(adj_amherst_k[gender_y==class_values[1],:],1)[nonzero_idx2]
                    
                    
                    count_of_zero_friends_in_khop0 = np.sum(gender_y==class_values[0]) - np.sum(nonzero_idx1) ## count of 0 issue
                    count_of_zero_friends_in_khop1 = np.sum(gender_y==class_values[1]) - np.sum(nonzero_idx2) ## count of 0 issue
                    
                    y_score = np.array(np.concatenate((1-mv_g1,mv_g2))).T[0] ## want classifier based on proportion from higher class; not proportion same, hence 1-mv_g1
                    y_test = np.concatenate((np.repeat(class_values[0],len(mv_g1)),
                                             np.repeat(class_values[1],len(mv_g2))))

                    auc_score = sklearn.metrics.roc_auc_score(label_binarize(y_test, np.unique(y_test)),
                                    y_score)
                    
                    writer.writerow( (tag, k, auc_score, count_of_zero_friends_in_khop0, count_of_zero_friends_in_khop1))
    file_output.close()
    print "Done!"
