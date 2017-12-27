from __future__ import division

# 12/2/2017
# KM Altenburger
# about: homophily index vs. Newman's assortativity

## a) run on soal: python soal_script_facebook_script_homophily_index_vs_Newmans_assortativity.py -i='/home/kaltenb/gender-graph/data' -o='.'
## a) run locally: cd Dropbox/gender_graph_data/manuscript/pnas/pnas_code/; python facebook_script_homophily_monophily.py -i='/Users/kristen/Dropbox/gender_graph_data/manuscript/code/fb_processing/data' -o='.'

folder_directory = '/home/kaltenb/gender-graph/NHB_revision_Nov2017/code' # main folder directory on SOAL
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
    
    file_output = open('../data/facebook_homophily_vs_newmans_assortativity_Dec2017.csv', 'wt')
    j =0
    writer = csv.writer(file_output)
    writer.writerow( ('school',
                      'cc_F_count', 'cc_M_count',
                      'cc_homophily_F', 'cc_homophily_M',
                      'cc_avg_homophily',
                      'cc_gender_assortativity'))
                      

    for f in listdir(args.input_dir):
        if f.endswith(args.file_ext):
            tag = f.replace(args.file_ext, '')
            j=j+1
            if (tag!='schools') and j<=3:
                print "Processing %s..." % tag
                input_file = path_join(args.input_dir, f)
                
                
                ## Descriptive Statistics on Raw, Original Data
                adj_matrix_tmp, metadata = parse_fb100_mat_file(input_file)

                gender_y_tmp = metadata[:,1] #gender
                #year_y_tmp = metadata[:,5] #year
                
                gender_dict = create_dict(range(len(gender_y_tmp)), gender_y_tmp)
                #year_dict = create_dict(range(len(year_y_tmp)), year_y_tmp)
          
                ## Compute Homophily/Monophily on Same Data Object Used for Prediction Setup
                # create corresponding y-/adj- objects
                (gender_y, adj_matrix_gender) = create_adj_membership(nx.from_scipy_sparse_matrix(adj_matrix_tmp),
                                                                      gender_dict,  # gender dictionary
                                                                      0,            # we drop nodes with gender_label = 0, missing
                                                                      'yes',        # yes to removing NA nodes
                                                                      0,            # set diagonal to 0, ie no self-loops
                                                                      None,         # for an undirected graph - we subset to nodes in largest connected component [ZGL]
                                                                      'gender')
                
                # gender-/class-year relative proportions
                proportion_gender = []
                block_size_gender = []
                avg_deg_gender = []
                
                class_labels = np.sort(np.unique(np.array(gender_y)))
                
                for i in range(len(np.unique(gender_y))):
                    block_size_gender.append( np.sum((gender_y==class_labels[i])+0))
                    proportion_gender.append( np.mean(gender_y==class_labels[i]))
                    avg_deg_gender.append(np.mean(np.array(np.sum(adj_matrix_gender,1))[gender_y==np.sort(np.unique(gender_y))[i]]))
                proportion_gender = np.array(proportion_gender)
 
                ## FB - homophily
                homophily_gender =  homophily_index_Jackson_alternative(adj_matrix_gender, gender_y) # observed homophily
                obs_homophily_F = homophily_gender[0]   # F - important assumes F label < M label
                obs_homophily_M = homophily_gender[1] # M - important assumes M label > F label
                
                G = nx.from_numpy_matrix(adj_matrix_gender)
                gender_dict_nx = create_dict(range(len(gender_y)), gender_y)
                nx.set_node_attributes(G, 'gender', gender_dict_nx)
                assortativity = nx.attribute_assortativity_coefficient(G, 'gender')
                
                writer.writerow( (tag,
                                   block_size_gender[0], block_size_gender[1],
                                  obs_homophily_F, obs_homophily_M,
                                  obs_homophily_F*np.mean(gender_y==class_labels[0])+obs_homophily_M*np.mean(gender_y==class_labels[1]),
                                  assortativity))

    file_output.close()
    print "Done!"
                

