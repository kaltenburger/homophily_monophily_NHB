# 1/12/2017
# KM Altenburger

# edited: 11/7/2017
# across FB100 - Wellesley/Smith/Vassar for the:
# a) full network and b) subsetted to largest connected component
# compute 1hop versus 2hop results


## a) run on soal: python facebook_script_1MV_2MV.py -i='/home/kaltenb/gender-graph/data' -o='.'
## a) run locally: python facebook_script_1MV_2MV.py -i='/Users/kristen/Dropbox/gender_graph_data/manuscript/code/fb_processing/data' -o='.'

## b) scp soal-1.stanford.edu:/home/kaltenb/gender-graph/code/pnas_code/facebook_output_LINK.csv ~



#folder_directory = '/home/kaltenb/gender-graph/NHB_revision_Nov2017/code' # main folder directory on SOAL
folder_directory =os.getcwd()

import os
import os
os.chdir(folder_directory)
execfile('./functions/python_libraries.py')
execfile('./functions/create_adjacency_matrix.py')
execfile('./functions/compute_homophily.py')
execfile('./functions/compute_monophily.py')
execfile('./functions/compute_monophily_beta_binom.py') ## 11/7/2017
execfile('./functions/compute_chi_square.py')
execfile('./functions/parsing.py')  # Sam Way's Code
execfile('./functions/mixing.py')   # Sam Way's Code
execfile('./functions/LINK.py')
execfile('./functions/majority_vote.py')


def interface():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input-dir', help='Input directory', required=True)
    args.add_argument('-e', '--file-ext', help='Input extension', default='.mat')
    args.add_argument('-o', '--output-dir', help='Output directory', required=True)
    args = args.parse_args()
    return args



if __name__=="__main__":
    args = interface()
    homophily_gender = []
    monophily_gender = []

    
    file_output = open('../data/facebook_output_majority_vote_AUC_accuracy_Nov2017.csv', 'wt')
    j =0
    writer = csv.writer(file_output)
    writer.writerow( ('school', 'percent_initially_unlabeled',
                      '1_MV_gender_mean_auc_wt', '1_MV_gender_se_auc_wt', '2_MV_gender_mean_auc_wt', '2_MV_gender_se_auc_wt',
                      '1_MV_gender_mean_auc_micro', '1_MV_gender_se_auc_micro', '2_MV_gender_mean_auc_micro', '2_MV_gender_se_auc_micro',
                      '1_MV_gender_mean_accuracy', '1_MV_gender_se_accuracy', '2_MV_gender_mean_accuracy', '2_MV_gender_se_accuracy',))
                      
    percent_initially_unlabelled = [0.5]
    percent_initially_labelled = np.subtract(1, percent_initially_unlabelled)

    for f in listdir(args.input_dir):
        if f.endswith(args.file_ext):
            tag = f.replace(args.file_ext, '')
            j=j+1
            if (tag!='schools'):
                print "Processing %s..." % tag
                input_file = path_join(args.input_dir, f)
                
                
                ## Descriptive Statistics on Raw, Original Data
                adj_matrix_tmp, metadata = parse_fb100_mat_file(input_file)

                gender_y_tmp = metadata[:,1] #gender
                gender_dict = create_dict(range(len(gender_y_tmp)), gender_y_tmp)
  
                ## Compute Homophily/Monophily on Same Data Object Used for Prediction Setup
                # create corresponding y-/adj- objects
                (gender_y, adj_matrix_gender) = create_adj_membership(nx.from_scipy_sparse_matrix(adj_matrix_tmp),
                                                                      gender_dict,  # gender dictionary
                                                                      0,            # we drop nodes with gender_label = 0, missing
                                                                      'yes',        # yes to removing
                                                                      0,            # set diagonal to 0, ie no self-loops
                                                                      None,         # for an undirected graph - we subset to nodes in largest connected component [ZGL]
                                                                      'gender')
                

                #1-hop MV
                (mean_accuracy_mv, se_accuracy_mv,
                 mean_micro_auc_mv_amherst,se_micro_auc_mv,
                 mean_wt_auc_mv,se_wt_auc_mv) =majority_vote_modified(percent_initially_unlabelled,
                                                                                       np.array(gender_y),
                                                                                       np.array(adj_matrix_gender),
                                                                                       num_iter=10)
                                                                                       
                                                                                       
                #2-hop MV
                adj2= np.matrix(adj_matrix_gender)**2
                adj2[range(adj2.shape[0]),range(adj2.shape[0])]=0
    
    
                (mean_accuracy_mv2, se_accuracy_mv2,
                mean_micro_auc_mv2,se_micro_auc_mv2,
                 mean_wt_auc_mv2,se_wt_auc_mv2) =majority_vote_modified(percent_initially_unlabelled,
                                                                               np.array(gender_y),
                                                                               np.array(adj2),
                                                                               num_iter=100)
        

                writer.writerow( (tag, percent_initially_labelled[0],
                                  mean_wt_auc_mv[0], se_wt_auc_mv[0],mean_wt_auc_mv2[0], se_wt_auc_mv2[0],
                                  mean_micro_auc_mv_amherst[0], se_micro_auc_mv[0],mean_micro_auc_mv2[0], se_micro_auc_mv2[0],
                                  mean_accuracy_mv[0], se_accuracy_mv[0],mean_accuracy_mv2[0], se_accuracy_mv2[0],))

    file_output.close()
    print "Done!"
                

