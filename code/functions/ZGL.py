from sklearn.metrics import confusion_matrix

## 4/24/2017
## about: ZGL function

#"Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions".  Xiaojin Zhu, Zoubin Ghahramani, John Lafferty. The Twentieth International Conference on Machine Learning (ICML-2003).
## note: helpful Matlab code that was adapted: http://pages.cs.wisc.edu/~jerryzhu/pub/harmonic_function.m
### assume: binary class set-up -- imbalanced classes -- allows k>2 classes; adjacency matrix -- assumed to be symmetric
def ZGL_finalized(feature_x, membership_y, num_unlabeled, num_iter, cv_setup=None, python_library='numpy'):
    percent_initially_labelled = np.subtract(1, num_unlabeled)

    ### ZGL CODE
    mean_accuracy = []
    se_accuracy = []

    mean_micro_auc = []
    se_micro_auc = []
    
    mean_wt_auc = []
    se_wt_auc = []

    if python_library == 'numpy':
        W_unordered = np.array(feature_x)
    if python_library == 'scipy':
        W_unordered = np.copy(feature_x)


    n = len(membership_y)
    classes = np.sort(np.unique(membership_y))
    class_labels = np.array(range(len(classes)))

    # relabel membership class labels - for coding convenience
    # preserve ordering of original class labels -- but force to be in sequential order now
    membership_y_update = np.copy(membership_y)
    for j in range(len(classes)):
        membership_y_update[membership_y_update == classes[j]] = class_labels[j]

    
    for i in range(len(percent_initially_labelled)):
        print(num_unlabeled[i])
        
        if cv_setup=='stratified':
            k_fold = cross_validation.StratifiedShuffleSplit((membership_y_update), n_iter=num_iter,
                                                             test_size=num_unlabeled[i],
                                                             random_state=0)
        else:
            k_fold = cross_validation.ShuffleSplit(len(membership_y_update), n_iter=num_iter,
                                                   test_size=num_unlabeled[i],
                                                   random_state=0)

        accuracy = []
        micro_auc = []
        wt_auc = []

        for k, (train, test) in enumerate(k_fold):
            #if k == 0:
                #print train
            idx = np.concatenate((train, test)) # concatenate train + test = L + U
            
            ## create W (4) in ZGL paper
            if python_library == 'numpy':
                W = np.reshape([W_unordered[row,col] for row in np.array(idx) for col in np.array(idx)],(n,n))
            
            if python_library == 'scipy':
                W_unordered = scipy.sparse.csc_matrix(feature_x)
                W = W_unordered[idx,:][:,idx]

            #fl: L*c label matrix from ZGL paper
            train_labels = np.array([np.array(membership_y_update)[id] for id in train]) # resort labels to be in same order as training data
            classes_train = np.sort(np.unique(train_labels))
            accuracy_score_benchmark = np.mean(np.array(train_labels) == np.max(class_labels))

            fl =np.array(np.matrix(label_binarize(train_labels,
                                  list(classes_train) + [np.max(classes_train)+1]))[:,0:(np.max(classes_train)+1)])


            # record testing gender labels for comparing predictions -- ie ground-truth
            true_test_labels = np.array([np.array(membership_y_update)[id] for id in test])
            classes_true_test = np.sort(np.unique(true_test_labels))
            ground_truth =np.array(np.matrix(label_binarize(true_test_labels,
                                                  list(classes_true_test) + [np.max(classes_true_test)+1]))[:,0:(np.max(classes_true_test)+1)])
            l = len(train) # number of labeled points
            u = len(test) # number of unlabeled points
            
            ## compute Equation (5) in ZGL paper
            W_ll = W[0:l,0:l]
            W_lu = W[0:l,l:(l+u)]
            W_ul = W[l:(l+u),0:l]
            W_uu = W[l:(l+u),l:(l+u)]

            if python_library == 'numpy':
                D = np.diag(np.sum(W, axis=1))
                D_ll = D[0:l,0:l]
                D_lu = D[0:l,l:(l+u)]
                D_ul = D[l:(l+u),0:l]
                D_uu = D[l:(l+u),l:(l+u)]
                harmonic_fxn =  np.dot(np.dot(np.linalg.inv(np.matrix(np.subtract(D_uu, W_uu))),np.matrix(W_ul)), np.matrix(fl))

            if python_library == 'scipy':
                D_tmp = scipy.sparse.csc_matrix.sum(W,1)
                udiag = np.zeros(n)
                ldiag = np.zeros(n)
                diag  = np.array(D_tmp.T)[0]
                D = scipy.sparse.csc_matrix(scipy.sparse.dia_matrix(([diag, udiag, ldiag], [0, 2, -2]), shape=(n, n)))
                D_ll = D[0:l,0:l]
                D_lu = D[0:l,l:(l+u)]
                D_ul = D[l:(l+u),0:l]
                D_uu = D[l:(l+u),l:(l+u)]
                harmonic_fxn =  scipy.sparse.csc_matrix.dot(scipy.sparse.csc_matrix.dot(scipy.sparse.linalg.inv(scipy.sparse.csc_matrix(scipy.subtract(D_uu,
                                                                                                                                                       W_uu))), W_ul),
                                            fl)
            harmonic_fxn_final = np.copy(harmonic_fxn)

            if len(np.unique(membership_y))>2:
                micro_auc.append(metrics.roc_auc_score(label_binarize(membership_y[test],np.unique(membership_y)),
                                                       harmonic_fxn_final,average='micro'))
                wt_auc.append(metrics.roc_auc_score(label_binarize(membership_y[test],np.unique(membership_y)),
                                                        harmonic_fxn_final,average='weighted'))
                accuracy.append(metrics.accuracy_score(label_binarize(membership_y[test],np.unique(membership_y)),
                                       harmonic_fxn_final))
            else:
                micro_auc.append(metrics.roc_auc_score(label_binarize(membership_y[test],np.unique(membership_y)),
                                           harmonic_fxn_final[:,1]-harmonic_fxn_final[:,0],average='micro'))
                wt_auc.append(metrics.roc_auc_score(label_binarize(membership_y[test],np.unique(membership_y)),
                                            harmonic_fxn_final[:,1]-harmonic_fxn_final[:,0],average='weighted'))
                                            
                y_true = label_binarize(membership_y[test],np.unique(membership_y))
                y_pred = ((harmonic_fxn_final[:,1]) >accuracy_score_benchmark)+0
                tn, fp, fn, tp = confusion_matrix(label_binarize(membership_y[test],np.unique(membership_y)),
                                                      ((harmonic_fxn_final[:,1]) >accuracy_score_benchmark)+0).ravel()
                                                      #accuracy.append((tn/(fp+tn)*0.5 + tp/(tp+fn))*0.5)

                accuracy.append(f1_score(y_true, y_pred, average='macro'))#, pos_label=1) )

#                accuracy.append(metrics.accuracy_score(label_binarize(membership_y[test],np.unique(membership_y)),
#                            (harmonic_fxn_final[:,1] > accuracy_score_benchmark)+0))

        mean_accuracy.append(np.mean(accuracy)) #placeholder
        se_accuracy.append(np.std(accuracy)) #placeholder
        
        mean_micro_auc.append(np.mean(micro_auc))
        se_micro_auc.append(np.std(micro_auc))
        mean_wt_auc.append(np.mean(wt_auc))
        se_wt_auc.append(np.std(wt_auc))
    return(mean_accuracy, se_accuracy, mean_micro_auc, se_micro_auc, mean_wt_auc,se_wt_auc)


