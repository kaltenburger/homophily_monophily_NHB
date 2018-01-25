# 4/24/2017
# about: relevant python libraries needed to run code
from __future__ import division
import argparse
from collections import Counter
import csv
#import igraph
import itertools
import json
import matplotlib.pyplot as plt
from mlxtend.classifier import EnsembleVoteClassifier
import networkx as nx
from numpy import save as np_save
import numpy as np
import pandas as pd
import pickle
import os
from os import listdir
from os.path import join as path_join
import random
import re
from rpy2.robjects.packages import importr
import rpy2
import rpy2.robjects as robjects
#import pandas.rpy.common as com
import rpy2.robjects.numpy2ri
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from scipy.stats import norm
from scipy.stats import bernoulli
import scipy
import scipy.special
import seaborn
seaborn.set_style(style='white')
import sklearn
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import label_binarize
from sklearn import cross_validation, datasets, linear_model
import statsmodels.api as sm
from scipy.stats import chisquare
import sys

## function to create + save dictionary of features
def create_dict(key, obj):
    return(dict([(key[i], obj[i]) for i in range(len(key)) ]))

def save_dict(feature_name, obj):
    with open(feature_name, 'w') as outfile:
        json.dump(obj, outfile)
