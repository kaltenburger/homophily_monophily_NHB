## Replication code: "Bias and variance in social network structure"


### Documentation

This repository contains all the correponding code to replicate the figures in "Bias and variance in social network structure". We provide links to the datasets (Facebook100, AddHealth, Political Blogs, and Noordin Top) in the data sub-folder.


### Directions

This repository set-up assumes that the FB100 (raw .mat files) and Add Health datasets have been acquired and are saved the data/original folder. Here are the directions:

1. Save raw files in data/original

2. Update file paths to your local directory settings in the following programs as indicated at the beginning of notebooks for setting file-paths.
    
3. Run code which is briefly described below:
   * 0_oSBM/ - includes notebooks for code related to simulations of oSBM graphs.
   * 1_analyze_FB100_AddHealth_Noordin_PolBlogs/ - includes all relevant code for data analysis presented in main paper and SI.
   * functions/compare_*.py - scripts for creating k-hop figures (AUC or proportion same).

All random number generators used in the analysis have been seeded deterministically to produce persistent cross-validation folds and thereby consistent results when re-running the analysis. The code for generating random graphs (sampled from the overdispersed stochastic block model) is not deterministically seeded. All code was written and tested for Python 2.7.12 with versions for the following main Python libraries:  networkx (1.9.1), numpy (1.13.3), pandas (0.20.3), rpy2 (2.8.5), sklearn (0.18.1), matplotlib(1.5.1), openpyxl (2.5.0), and xlrd (1.1.0). The code has know incompatibilities with Python 3.x and with networkx 2.x.

### Authors
* Kristen M. Altenburger, kaltenb@stanford.edu
* Johan Ugander, jugander@stanford.edu
