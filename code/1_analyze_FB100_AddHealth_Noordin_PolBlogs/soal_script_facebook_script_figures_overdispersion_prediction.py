from __future__ import division

# edited: 11/7/2017 [original + beta-binomial-based monophily measures]
# originally: 12/17/2016
# KM Altenburger

# across FB100 - Wellesley/Smith/Vassar for the:
# a) full network and b) subsetted to largest connected component
# compute descriptive stats, homophily, monophily metrics

# Homophily = E(d_F/d) for F and E(d_M/d) for M
# Monophily = Var(d_F/d) for F and Var(d_M/d) for M

## a) run on soal: python soal_script_facebook_script_figures_overdispersion_prediction.py -i='/home/kaltenb/gender-graph/data' -o='.'
## a) run locally: cd Dropbox/gender_graph_data/manuscript/pnas/pnas_code/; python facebook_script_homophily_monophily.py -i='/Users/kristen/Dropbox/gender_graph_data/manuscript/code/fb_processing/data' -o='.'


## b) how to move file from soal to corn: starting on corn - scp soal-1.stanford.edu:/home/kaltenb/gender-graph/code/pnas_output_data/facebook_homophily_monophily_output_updated.csv ~

folder_directory = '/home/kaltenb/gender-graph/NHB_revision_Nov2017/code' # main folder directory on SOAL
import os
os.chdir(folder_directory)
execfile('./functions/python_libraries.py')
execfile('./functions/create_adjacency_matrix.py')
execfile('./functions/compute_homophily.py')
execfile('./functions/compute_monophily.py')
execfile('./functions/majority_vote.py')
execfile('./functions/ZGL.py')
execfile('./functions/LINK_finalized.py')
execfile('./functions/compute_monophily_beta_binom.py') ## 11/7/2017
execfile('./functions/compute_chi_square.py')
execfile('./functions/parsing.py')  # Sam Way's Code
execfile('./functions/mixing.py')   # Sam Way's Code
execfile('./functions/benchmark_classifier.py')
def interface():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input-dir', help='Input directory', required=True)
    args.add_argument('-e', '--file-ext', help='Input extension', default='.mat')
    args.add_argument('-o', '--output-dir', help='Output directory', required=True)
    args = args.parse_args()
    return args



if __name__=="__main__":
    args = interface()
    
    j =0


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
                
                F_fb_label = 1
                M_fb_label = 2
                #F
                in_F_degree = adj_matrix_gender[gender_y==F_fb_label,] * np.matrix((gender_y==F_fb_label)+0).T
                total_F_degree = np.sum(adj_matrix_gender[gender_y==F_fb_label,] ,1)
                h_F = np.mean(in_F_degree)/np.mean(total_F_degree)

                #M
                in_M_degree = adj_matrix_gender[gender_y==M_fb_label,] * np.matrix((gender_y==M_fb_label)+0).T
                total_M_degree = np.sum(adj_matrix_gender[gender_y==M_fb_label,] ,1)
                h_M = np.mean(in_M_degree)/np.mean(total_M_degree)


                n_iter = 100000
                d_iF = map(np.int,np.array(total_F_degree.T)[0])

                mc_F = []
                for j in range(n_iter):
                    mc_F.append( np.random.binomial(n=map(np.int,np.array(total_F_degree.T)[0]), p=h_F)/d_iF)

                mc_F_final = np.array(mc_F).flatten()


                d_iM = map(np.int,np.array(total_M_degree.T)[0])
                mc_M = []
                for j in range(n_iter):
                    mc_M.append( np.random.binomial(n=map(np.int,np.array(total_M_degree.T)[0]), p=h_M)/d_iM)

                mc_M_final = np.array(mc_M).flatten()
                    
                alpha_empirical = 0.4

#%matplotlib inline
                nbins = 45
                f, (ax1, ax2) = plt.subplots(1, 2,
                                             sharey=False, sharex=False,
                                             figsize=(8, 3))
                plt.setp(ax1, xticks=[0,0.25, 0.5, 0.75, 1], xticklabels=['0', '0.25', '0.50', '0.75', '1'])
                ax1.set_xticks([0,0.25, 0.5, 0.75, 1])
                ax1.minorticks_on()
                ax1.tick_params('both', length=4, width=1, which='major', left=1, bottom=1, top=0, right=0)


                ax1.hist(in_F_degree/total_F_degree,
                         bins = np.linspace(0,1,nbins),
                         normed = True,
                         edgecolor = 'white',
                         color='red',alpha=0.25)

                ax1.hist(mc_F_final,
                         bins = np.linspace(0,1,nbins),
                         normed = True,
                         edgecolor = 'red',histtype='step',
                         color='white',alpha=1,lw=1.5)

                ax1.set_ylabel('Normalized Frequency')
                ax1.set_xlabel('Proportion of $in$-class neighbors, $d_{i,\\mathrm{in}}/d_i$')
                ax1.spines["right"].set_visible(False)
                ax1.spines["top"].set_visible(False)
                ax1.set_ylim(0,8)
                ax1.set_xlim(0,1.01)

                ax2.set_ylim(0,8)
                ax2.set_xlim(0,1.01)
                plt.setp(ax2, xticks=[0,0.25, 0.5, 0.75, 1], xticklabels=['0', '0.25', '0.50', '0.75', '1'])
                ax2.set_xticks([0,0.25, 0.5, 0.75, 1])
                ax2.minorticks_on()
                ax2.tick_params('both', length=4, width=1, which='major', left=1, bottom=1, top=0, right=0)


                ax2.hist(in_M_degree/total_M_degree,
                         bins = np.linspace(0,1,nbins),
                         normed = True,
                         edgecolor = 'white',
                         color='black',alpha=0.25)
                ax2.hist(mc_M_final,
                         bins = np.linspace(0,1,nbins),
                         normed = True,
                         edgecolor = 'black',histtype='step',
                         color='white',alpha=1,lw=1.5)
                ax2.spines["right"].set_visible(False)
                ax2.spines["top"].set_visible(False)
                ax2.set_xlabel('Proportion of $in$-class Neighbors, $d_{i,\\mathrm{in}}/d_i$')
                ax2.set_ylabel('Normalized Frequency')

                ax1.annotate('Females Null', xy=(0.05,7.5),
                             color='red', alpha=1, size=12)
                ax1.annotate('Females Empirical', xy=(0.05,7),
                             color='red', alpha=alpha_empirical, size=12)
                ax2.annotate('Males Null', xy=(0.05,7.5),
                             color='black', alpha=1, size=12)
                ax2.annotate('Males Empirical', xy=(0.05,7),
                             color='black', alpha=alpha_empirical, size=12)
                #plt.tight_layout()
                #plt.title('test')
                pp = PdfPages('../figs10/Facebook_' + tag + '_overdispersion_100k.pdf')
                pp.savefig()
                pp.close()
                
                
                n_iter = 10
                percent_initially_unlabelled = [0.99,0.95,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.01]
                percent_initially_labelled = np.subtract(1, percent_initially_unlabelled)
                    
                (mean_accuracy_amherst, se_accuracy_amherst,
                 mean_micro_auc_amherst,se_micro_auc_amherst, mean_wt_auc_amherst_lbfgs,se_wt_auc_amherst)= LINK(percent_initially_unlabelled, ## note: mean_se_model assumes a vector of x% initially labeled
                                                                                                                 np.array(gender_y), ## gender labels
                                                                                                                 np.matrix(adj_matrix_gender), ## adjacency matrix
                                                                                                                 clf = linear_model.LogisticRegression(penalty='l2',
                                                                                                                                                       solver='lbfgs',
                                                                                                                                                       C=10e20),num_iter=n_iter)
                                                                                                                     
                (mean_accuracy_mv_amherst, se_accuracy_mv_amherst,
                 mean_micro_auc_mv_amherst,se_micro_auc_mv_amherst,
                 mean_wt_auc_mv_amherst,se_wt_auc_mv_amherst) =majority_vote_finalized(percent_initially_unlabelled,
                                                                                       np.array(gender_y),
                                                                                       np.array(adj_matrix_gender),
                                                                                       num_iter=n_iter)
                adj_amherst2= np.matrix(adj_matrix_gender)**2
                adj_amherst2[range(adj_amherst2.shape[0]),range(adj_amherst2.shape[0])]=0 ## remove self-loops

                (mean_accuracy_mv2_amherst2, se_accuracy_mv2_amherst2,
                 mean_micro_auc_mv2_amherst2,se_micro_auc_mv2_amherst2,
                 mean_wt_auc_mv2_amherst2,se_wt_auc_mv2_amherst2) =majority_vote_finalized(percent_initially_unlabelled,
                                                                                           np.array(gender_y),
                                                                                           np.array(adj_amherst2),
                                                                                           num_iter=n_iter)
    
                (mean_accuracy_zgl_amherst, se_accuracy_zgl_amherst,
                 mean_micro_auc_zgl_amherst,se_micro_auc_zgl_amherst,
                 mean_wt_auc_zgl_amherst,se_wt_auc_zgl_amherst) =ZGL_finalized(np.array(adj_matrix_gender),
                                                                               np.array(gender_y),percent_initially_unlabelled,
                                                                               num_iter=n_iter)
                    
                (mean_wt_auc_baseline_amherst,se_wt_auc_baseline_amherst) = random_classifier(np.array(adj_matrix_gender),
                                                                              np.array(gender_y),
                                                                              percent_initially_unlabelled,
                                                                              num_iter=n_iter)


#%matplotlib inline
                from matplotlib.ticker import FixedLocator,LinearLocator,MultipleLocator, FormatStrFormatter

                fig = plt.figure()
                seaborn.set_style(style='white')
                from mpl_toolkits.axes_grid1 import Grid
                grid = Grid(fig, rect=111, nrows_ncols=(1,1),
                            axes_pad=0.1, label_mode='L')
                for i in range(4):
                    if i == 0:
                        grid[i].xaxis.set_major_locator(FixedLocator([0,25,50,75,100]))
                        grid[i].yaxis.set_major_locator(FixedLocator([0.4, 0.5,0.6,0.7,0.8,0.9,1]))
                        
                        grid[i].errorbar(percent_initially_labelled*100, mean_wt_auc_amherst_lbfgs,
                                         yerr=se_wt_auc_amherst, fmt='--o', capthick=2,
                                         alpha=1, elinewidth=3, color='black')
                        grid[i].errorbar(percent_initially_labelled*100, mean_wt_auc_zgl_amherst,
                                                          yerr=se_micro_auc_zgl_amherst, fmt='--o', capthick=2,
                                                          alpha=1, elinewidth=3, color='orange')
                        grid[i].errorbar(percent_initially_labelled*100, mean_wt_auc_mv_amherst,
                                                          yerr=se_micro_auc_mv_amherst, fmt='--o', capthick=2,
                                                          alpha=1, elinewidth=3, color='red')
                        grid[i].errorbar(percent_initially_labelled*100, mean_wt_auc_mv2_amherst2,
                                                          yerr=se_micro_auc_mv2_amherst2, fmt='--o', capthick=2,
                                                          alpha=1, elinewidth=3, color='maroon')
                        grid[i].errorbar(percent_initially_labelled*100, mean_wt_auc_baseline_amherst,
                                                          yerr=se_wt_auc_baseline_amherst, fmt='--o', capthick=2,
                                                          alpha=1, elinewidth=3, color='gray')
                        grid[i].set_ylim(0.3,1.1)
                        grid[i].set_xlim(0,101)
                        grid[i].annotate('LINK', xy=(3, 0.99),
                                                          color='black', alpha=1, size=12)
                        grid[i].annotate('2-hop MV', xy=(3, 0.96),
                                                          color='maroon', alpha=1, size=12)
                        grid[i].annotate('1-hop MV', xy=(3, 0.93),
                                                          color='red', alpha=1, size=12)
                        grid[i].annotate('ZGL', xy=(3, 0.90),
                                                          color='orange', alpha=1, size=12)
                        grid[i].annotate('Baseline', xy=(3, 0.87),
                                                          color='gray', alpha=1, size=12)
                        grid[i].set_ylim(0.49,1.01)
                        grid[i].set_xlim(0,100)
                        grid[i].spines['right'].set_visible(False)
                        grid[i].spines['top'].set_visible(False)
                        grid[i].tick_params(axis='both', which='major', labelsize=13)
                        grid[i].tick_params(axis='both', which='minor', labelsize=13)
                        grid[i].set_xlabel('Percent of Nodes Initially Labeled')
                        grid[i].set_ylabel('AUC')

                plt.setp(ax1, xticks=[0,25, 50, 75, 100], xticklabels=['0', '25', '50', '75', '100'])
                grid[0].set_xticks([0,25, 50, 75, 100])
                grid[0].set_yticks([ 0.4, 0.5,0.6,0.7,0.8,0.9,1])
                grid[0].minorticks_on()
                grid[0].tick_params('both', length=4, width=1, which='major', left=1, bottom=1, top=0, right=0)
                pp = PdfPages('../figs10/'+tag +'_College_Inference.pdf')
                pp.savefig()
                pp.close()

    file_output.close()
    print "Done!"
                

