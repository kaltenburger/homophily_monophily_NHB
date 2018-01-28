from __future__ import division
## about: process homophily/monophily across all Add Health Schools [in-directed]


## for reference: [Male == 1; Female == 2, Unreported/Missing = 0]
## codings are based on: http://moreno.ss.uci.edu/data.html#adhealth

## how to run [example]:
## cd [Folder_Path_of_Script]
## python c_add_health_script_homophily_monophily_directed_in.py -i=[Folder_Path_of_Converted_GML] -o=[Folder_Path_of_Output_CSV]

import os
folder_directory =os.getcwd()

os.chdir(folder_directory)
execfile('../functions/python_libraries.py')
execfile('../functions/compute_homophily.py')
execfile('../functions/compute_monophily.py')
execfile('../functions/compute_monophily_beta_binom.py') ## 11/9/2017
execfile('../functions/compute_chi_square.py')
execfile('../functions/parsing.py')  # Sam Way's Code
execfile('../functions/mixing.py')   # Sam Way's Code
execfile('../functions/create_directed_adjacency_matrix.py')


def interface():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input-dir', help='Input directory', required=True)
    args.add_argument('-e', '--file-ext', help='Input extension', default='.gml') # KM - files already converted to .gml - find this code
    args.add_argument('-o', '--output-dir', help='Output directory', required=True)
    args = args.parse_args()
    return args



if __name__=="__main__":
    args = interface()
    homophily_gender = []
    monophily_gender = []

    
    file_output = open('../../data/output/add_health_output_out_directed_links_homophily_monophily_NatureHB_Nov2017.csv', 'wt') # change file name to directed
    j =0
    writer = csv.writer(file_output)
    writer.writerow( ('school', 'raw_F_count', 'raw_M_count', 'raw_?_count',
                      'cc_F_count', 'cc_M_count', 'ratio_F',
                      'cc_average_degree_F', 'cc_average_degree_M','cc_max_deg_F','cc_max_deg_M',
                      'cc_homophily_F', 'cc_homophily_M',
                      'b0_glm_F','b0_dispmod_glm_F', 'b0_glm_M','b0_dispmod_glm_M',
                      'cc_monophily_F', 'cc_monophily_M',
                      'chi_square_p_value_F', 'chi_square_p_value_M',
                      'cc_monophily_F_beta_binom','cc_monophily_M_beta_binom'))

    os.chdir('/Users/kristen/Dropbox/gender_graph_data/add-health/converted_gml/')
    for f in listdir(args.input_dir):
        if f.endswith(args.file_ext):
            tag = f.replace(args.file_ext, '')
            j=j+1

            id = re.findall(r'\d+', f)
            print "Processing %s..." % f


            # updated for directed version of graph
            ah_graph = nx.read_gml(f)
 
 
            #out-link
            (ah_gender_out, adj_gender_out) = create_directed_adj_membership(ah_graph,
                                                                nx.get_node_attributes(ah_graph, 'comm' + str(id[0]) +'sex'),                                                                    0,
                                                                   'yes',
                                                                   0,
                                                                   'out', 
                                                                   'gender')
        


                                                                    

            ## Descriptive Statistics on Raw, Original Data
            gender_y_tmp = nx.get_node_attributes(ah_graph, 'comm' + str(id[0]) +'sex')
            
            # Original Data
            raw_gender_F_undirected = np.sum((np.array(gender_y_tmp.values())==2)+0)
            raw_gender_M_undirected = np.sum((np.array(gender_y_tmp.values())==1)+0)
            raw_gender_unknown_undirected = np.sum((np.array(gender_y_tmp.values())==0)+0)

            
            # gender-/class-year relative proportions
            proportion_gender = []
            block_size_gender = []
            avg_deg_gender = []
            max_deg = []
            class_labels = np.sort(np.unique(np.array(ah_gender_out)))
            for i in range(len(class_labels)):
                block_size_gender.append( np.sum((ah_gender_out==class_labels[i])+0))
                proportion_gender.append( np.mean(ah_gender_out==class_labels[i]))
                avg_deg_gender.append(np.mean(np.array(np.sum(adj_gender_out,1))[ah_gender_out==class_labels[i]]))
                max_deg.append(np.max(np.array(np.sum(adj_gender_out,1))[ah_gender_out==class_labels[i]]))
            proportion_gender = np.array(proportion_gender)
            avg_deg_gender = np.array(avg_deg_gender)
            block_size_gender = np.array(block_size_gender)
            max_deg = np.array(max_deg)


            if len(block_size_gender) >= 2:
                ## AH - homophily
                homophily_gender =  homophily_index_Jackson_alternative(adj_gender_out, ah_gender_out) # observed homophily
                obs_homophily_F = homophily_gender[1]   # F - important assumes F label < M label
                obs_homophily_M = homophily_gender[0] # M - important assumes M label > F label
                
                
                ## compare with b0 terms
                b0_temp = np.exp(monophily_index_overdispersion_Williams_with_intercept(np.matrix(adj_gender_out), np.array(ah_gender_out)))
                b0_glm_F = (b0_temp/(1+b0_temp))[:,1][0] # F
                b0_glm_M = (b0_temp/(1+b0_temp))[:,0][0] # M
                
                b0_dispmod_glm_F = (b0_temp/(1+b0_temp))[:,1][1] # F
                b0_dispmod_glm_M = (b0_temp/(1+b0_temp))[:,0][1] # M
                
                
                ## AH - monophily
                monophily_gender = monophily_index_overdispersion_Williams(adj_gender_out, ah_gender_out)
                obs_monophily_F = np.float(monophily_gender[1])  # F - important assumes F label < M label
                obs_monophily_M = np.float(monophily_gender[0]) # M - important assumes M label > F label

                chi_square_p_value_gender = compute_chi_square_statistic(np.matrix(adj_gender_out), np.array(ah_gender_out))
                chi_square_p_value_F = np.float(chi_square_p_value_gender[1])
                chi_square_p_value_M = np.float(chi_square_p_value_gender[0])

                ## AH - monophily, beta-binom version
                ## added 11/9/2017
                monophily_gender_bb = monophily_index_beta_bin(adj_gender_out, np.array(ah_gender_out))
                obs_monophily_F_bb = np.float(monophily_gender_bb[1])  # F - important assumes F label < M label
                obs_monophily_M_bb = np.float(monophily_gender_bb[0]) # M - important assumes M label > F label
    
    

                block_F =block_size_gender[1]
                block_M =block_size_gender[0]
                avg_deg_F = avg_deg_gender[1]
                avg_deg_M = avg_deg_gender[0]
                max_deg_F = max_deg[1]
                max_deg_M = max_deg[0]
                prop_F =block_F/(block_F+block_M)
        
            else:
                ## AH - homophily
                homophily_gender =  ''
                obs_homophily_F = ''
                obs_homophily_M = ''
                
                b0_glm_F = ''
                b0_dispmod_glm_F= ''
                b0_glm_M= ''
                b0_dispmod_glm_M= ''
            
            
                ## AH - monophily - bb
                obs_monophily_F_bb = ''
                obs_monophily_M_bb = ''
                ## AH - monophily
                monophily_gender = ''
                obs_monophily_F = ''
                obs_monophily_M = ''
                block_F =''
                block_M =''
                avg_deg_F = ''
                avg_deg_M = ''
                prop_F = ''
                chi_square_p_value_F= ''
                chi_square_p_value_M = ''


            writer.writerow( (tag, raw_gender_F_undirected,raw_gender_M_undirected, raw_gender_unknown_undirected,
                  block_F, block_M,prop_F,
                  avg_deg_F, avg_deg_M,max_deg_F,max_deg_M,
                  obs_homophily_F, obs_homophily_M,
                  b0_glm_F,b0_dispmod_glm_F, b0_glm_M, b0_dispmod_glm_M,
                  obs_monophily_F,obs_monophily_M,
                  chi_square_p_value_F, chi_square_p_value_M,
                              obs_monophily_F_bb, obs_monophily_M_bb))



    file_output.close()
    print "Done!"
