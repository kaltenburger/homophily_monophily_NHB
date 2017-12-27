from __future__ import division
import os
## about: compute homophily/monophily across FB100 dataset

## how to run:
## cd /Users/kristen/Documents/gender_graph_code/code/0_analyze_FB100_AddHealth/
## python c_facebook_script_homophily_monophily.py -i='/Users/kristen/Dropbox/gender_graph_data/manuscript/code/fb_processing/data' -o='/Users/kristen/Documents/gender_graph_code/data/'

## how to run on soal
## OLD: python compare_k_hop_friends_vs_proportion_class_same.py -i='/home/kaltenb/gender-graph/data' -o='/home/kaltenb/gender-graph'


# cd /Users/kristen/Dropbox/gender_graph_data/manuscript/nature_hb/3_nature_hb_running_list_of_revisions_post_may_2017/code
## CURRENT: python c_facebook_script_homophily_monophily.py -i='' -o='/Users/kristen/Dropbox/gender_graph_data/manuscript/nature_hb/3_nature_hb_running_list_of_revisions_post_may_2017/code'

import os
folder_directory =os.getcwd()
#folder_directory = '/Users/kristen/Dropbox/gender_graph_data/manuscript/nature_hb/gender_graph_final_code_NatureHB/code/1_analyze_FB100_AddHealth_Noordin_PolBlogs/'
os.chdir(folder_directory)

code_path = '/home/kaltenb/gender-graph/data/'

execfile('./functions/python_libraries.py')
execfile('./functions/create_adjacency_matrix.py')
execfile('./functions/parsing.py')  # Sam Way's Code
execfile('./functions/mixing.py')   # Sam Way's Code


def interface():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input-dir', help='Input directory', required=True)
    args.add_argument('-e', '--file-ext', help='Input extension', default='.mat')
    args.add_argument('-o', '--output-dir', help='Output directory', required=True)
    args = args.parse_args()
    return args

if __name__=="__main__":
    args = interface()
    #print args
    homophily_gender = []
    monophily_gender = []
    
    file_output = open('/home/kaltenb/gender-graph/NHB_revision_Nov2017/data/khop_vs_proportion_same_Nov2017.csv', 'wt')
    j =0
    writer = csv.writer(file_output)
    writer.writerow( ('school', 'k_hop', 'proportion_nodes_majority_same_class_in_khop_neighborhood', 'count_0_friends_khop_class1','count_0_friends_khop_class2'))
                      

    for f in listdir(code_path):
        if f.endswith(args.file_ext):
            tag = f.replace(args.file_ext, '')
            #print tag
            j=j+1
            if (tag!='schools') and (tag=='MIT8'):
                print "Processing %s..." % tag
                input_file = path_join(code_path, f)
                
                ## Descriptive Statistics on Raw, Original Data
                adj_matrix_tmp, metadata = parse_fb100_mat_file(input_file)
                gender_y_tmp = metadata[:,1] #gender
                gender_dict = create_dict(range(len(gender_y_tmp)), gender_y_tmp)

  
                ## Compute Homophily/Monophily on Same Data Object Used for Prediction Setup
                ## create corresponding y-/adj- objects
                (gender_y, adj_matrix_gender) = create_adj_membership(nx.from_scipy_sparse_matrix(adj_matrix_tmp),
                                                                      gender_dict,  # gender dictionary
                                                                      0,            # we drop nodes with gender_label = 0, missing
                                                                      'yes',        # yes to removing NA nodes
                                                                      0,            # set diagonal to 0, ie no self-loops
                                                                      None,         # for an undirected graph - we subset to nodes in largest connected component [ZGL]
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
                    
                    
                    proportion_majority_same = np.mean(np.concatenate((np.array(mv_g1).T[0]>np.mean(gender_y==class_values[0]), ## proportion of class 1 with more same class neighbors than baseline proportion class 1
                                                                       np.array(mv_g2).T[0]>np.mean(gender_y==class_values[1])))) ## proportion of class 2 with more same class neighbors than baseline proportion class 2
                                                                       # then np.mean(.) computes overall proportion across all nodes

                    count_of_zero_friends_in_khop0 = np.sum(gender_y==class_values[0]) - np.sum(nonzero_idx1) ## count of 0 issue
                    count_of_zero_friends_in_khop1 = np.sum(gender_y==class_values[1]) - np.sum(nonzero_idx2) ## count of 0 issue
                                                                       
                    writer.writerow( (tag, k, proportion_majority_same, count_of_zero_friends_in_khop0, count_of_zero_friends_in_khop1))
    file_output.close()
    print "Done!"
