from __future__ import division

# edited: 11/7/2017 [original + beta-binomial-based monophily measures]
# originally: 12/17/2016
# KM Altenburger

# across FB100 - Wellesley/Smith/Vassar for the:
# a) full network and b) subsetted to largest connected component
# compute descriptive stats, homophily, monophily metrics

# Homophily = E(d_F/d) for F and E(d_M/d) for M
# Monophily = Var(d_F/d) for F and Var(d_M/d) for M

## a) run on soal: python facebook_script_homophily_monophily.py -i='/home/kaltenb/gender-graph/data' -o='.'
## a) run locally: cd Dropbox/gender_graph_data/manuscript/pnas/pnas_code/; python facebook_script_homophily_monophily.py -i='/Users/kristen/Dropbox/gender_graph_data/manuscript/code/fb_processing/data' -o='.'


## b) how to move file from soal to corn: starting on corn - scp soal-1.stanford.edu:/home/kaltenb/gender-graph/code/pnas_output_data/facebook_homophily_monophily_output_updated.csv ~

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
    
    file_output = open('../data/facebook_homophily_monophily_output_Williams_BetaBinom_Nov2017.csv', 'wt')
    j =0
    writer = csv.writer(file_output)
    writer.writerow( ('school', 'raw_F_count', 'raw_M_count', 'raw_?_count',
                      'cc_F_count', 'cc_M_count', 'ratio_F',
                      'cc_average_degree_F', 'cc_average_degree_M',
                      'cc_homophily_F', 'cc_homophily_M',
                      'cc_homophily_conf_int_l_99_F','cc_homophily_conf_int_u_99_F', ## 99% interval - F
                      'cc_homophily_conf_int_l_99_M','cc_homophily_conf_int_u_99_M', ## .. - M
                      'cc_homophily_conf_int_l_99_9_F','cc_homophily_conf_int_u_99_9_F', ## 99.9% interval - F
                      'cc_homophily_conf_int_l_99_9_M','cc_homophily_conf_int_u_99_9_M', ## .. - M
                      'b0_glm_F','b0_dispmod_glm_F', 'b0_glm_M','b0_dispmod_glm_M',
                      'cc_monophily_F', 'cc_monophily_M',
                      'chi_square_p_value_F', 'chi_square_p_value_M',
                      'cc_monophily_F_beta_binom','cc_monophily_M_beta_binom'))
                      

    for f in listdir(args.input_dir):
        if f.endswith(args.file_ext):
            tag = f.replace(args.file_ext, '')
            j=j+1
            if (tag!='schools'):# and j<=3:
                print "Processing %s..." % tag
                input_file = path_join(args.input_dir, f)
                
                
                ## Descriptive Statistics on Raw, Original Data
                adj_matrix_tmp, metadata = parse_fb100_mat_file(input_file)

                gender_y_tmp = metadata[:,1] #gender
                #year_y_tmp = metadata[:,5] #year
                
                gender_dict = create_dict(range(len(gender_y_tmp)), gender_y_tmp)
                #year_dict = create_dict(range(len(year_y_tmp)), year_y_tmp)
                
                
                raw_gender_F = np.sum((gender_y_tmp==1)+0)
                raw_gender_M = np.sum((gender_y_tmp==2)+0)
                raw_gender_unknown = np.sum((gender_y_tmp==0)+0)
  
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
                avg_deg_gender = np.array(avg_deg_gender)
                block_size_gender = np.array(block_size_gender)


                ## FB - homophily
                homophily_gender =  homophily_index_Jackson_alternative(adj_matrix_gender, gender_y) # observed homophily
                obs_homophily_F = homophily_gender[0]   # F - important assumes F label < M label
                obs_homophily_M = homophily_gender[1] # M - important assumes M label > F label
                
                
                homophily_significance = np.exp(monophily_index_overdispersion_Williams_with_intercept_SE_99(adj_matrix_gender, gender_y))/(1+np.exp(monophily_index_overdispersion_Williams_with_intercept_SE_99(adj_matrix_gender, gender_y)))
                cc_homophily_conf_int_l_99_F = homophily_significance[:,0][0]
                cc_homophily_conf_int_u_99_F = homophily_significance[:,0][1]

                cc_homophily_conf_int_l_99_M = homophily_significance[:,1][0]
                cc_homophily_conf_int_u_99_M = homophily_significance[:,1][1]
                
                homophily_significance = np.exp(monophily_index_overdispersion_Williams_with_intercept_SE_99_9(adj_matrix_gender, gender_y))/(1+np.exp(monophily_index_overdispersion_Williams_with_intercept_SE_99_9(adj_matrix_gender, gender_y)))
                cc_homophily_conf_int_l_99_9_F = homophily_significance[:,0][0]
                cc_homophily_conf_int_u_99_9_F = homophily_significance[:,0][1]
                
                cc_homophily_conf_int_l_99_9_M = homophily_significance[:,1][0]
                cc_homophily_conf_int_u_99_9_M = homophily_significance[:,1][1]
                
    
                b0_temp = np.exp(monophily_index_overdispersion_Williams_with_intercept(np.matrix(adj_matrix_gender), np.array(gender_y)))
                #print b0_temp
                b0_glm_F = (b0_temp/(1+b0_temp))[:,0][0] # F
                b0_glm_M = (b0_temp/(1+b0_temp))[:,1][0] # M

                b0_dispmod_glm_F = (b0_temp/(1+b0_temp))[:,0][1] # F
                b0_dispmod_glm_M = (b0_temp/(1+b0_temp))[:,1][1] # M


                ## FB - monophily
                monophily_gender = monophily_index_overdispersion_Williams(adj_matrix_gender, gender_y)
                obs_monophily_F = np.float(monophily_gender[0])  # F - important assumes F label < M label
                obs_monophily_M = np.float(monophily_gender[1]) # M - important assumes M label > F label
                

                ## FB - beta-binomial monophily
                monophily_gender_bb = monophily_index_beta_bin(adj_matrix_gender, gender_y)
                obs_monophily_F_bb = np.float(monophily_gender_bb[0])  # F - important assumes F label < M label
                obs_monophily_M_bb = np.float(monophily_gender_bb[1]) # M - important assumes M label > F label
                

                chi_square_p_value_gender = compute_chi_square_statistic(np.matrix(adj_matrix_gender), np.array(gender_y))
                chi_square_p_value_F = np.float(chi_square_p_value_gender[0])
                chi_square_p_value_M = np.float(chi_square_p_value_gender[1])
                writer.writerow( (tag, raw_gender_F,raw_gender_M, raw_gender_unknown,
                                   block_size_gender[0], block_size_gender[1], block_size_gender[0]/(block_size_gender[0]+block_size_gender[1]),
                                  avg_deg_gender[0], avg_deg_gender[1],
                                  obs_homophily_F, obs_homophily_M,
                                  cc_homophily_conf_int_l_99_F,cc_homophily_conf_int_u_99_F,
                                  cc_homophily_conf_int_l_99_M,cc_homophily_conf_int_u_99_M,
                                  cc_homophily_conf_int_l_99_9_F,cc_homophily_conf_int_u_99_9_F,
                                  cc_homophily_conf_int_l_99_9_M,cc_homophily_conf_int_u_99_9_M,
                                  b0_glm_F,b0_dispmod_glm_F, b0_glm_M, b0_dispmod_glm_M,
                                  obs_monophily_F,obs_monophily_M,
                                  chi_square_p_value_F, chi_square_p_value_M,
                                  obs_monophily_F_bb, obs_monophily_M_bb))

    file_output.close()
    print "Done!"
                

