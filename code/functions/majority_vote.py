## 4/24/2017
## edited: 11/4/2017 - accuracy scores based on relative class proportions
## about: majority vote classifiers.

## notes:
## when a node has 0 labeled friends which can occur in the sparesely labeled case -- assign proportional to training sample classes


def majority_vote_finalized(num_unlabeled, membership_y,feature_x, num_iter, cv_setup=None):
    num_labels = len(np.unique(np.array(membership_y)))
    class_labels = np.sort(np.unique(np.array(membership_y))) #unique label IDs
    class_preference = []
    mean_accuracy = []
    se_accuracy = []
    mean_micro_auc = []
    se_micro_auc = []
    mean_wt_auc = []
    se_wt_auc = []

    for i in range(len(num_unlabeled)):
        print(num_unlabeled[i])
        accuracy = []
        micro_auc = []
        wt_auc = []
        
        
        ## cross-validation set-up
        if cv_setup=='stratified':
            k_fold = cross_validation.StratifiedShuffleSplit((membership_y), n_iter=num_iter,
                                                             test_size=num_unlabeled[i],
                                                             random_state=0)
        else:
            k_fold = cross_validation.ShuffleSplit(len(membership_y), n_iter=num_iter,
                                                   test_size=num_unlabeled[i],
                                                   random_state=0)
        for k, (train, test) in enumerate(k_fold):
            #if k ==0:
            #print train
            labeled_data = np.copy(np.array(membership_y))
            ground_truth_testing = np.array(labeled_data)[test]
            labeled_data[test]=np.max(class_labels)+1 # ignore testing labels -- don't have access as part of training -- want to assing test label outside of possible training labels
            proportion_class_preference = []
            for m in range(num_labels):
                proportion_class_preference.append(feature_x * np.matrix((labeled_data==class_labels[m])+0).T/(feature_x * np.matrix((labeled_data!=(np.max(class_labels)+1))+0).T)) ## order of proportions is class_labels[0], class_labels[1], etc.
            preference_by_class_matrix = np.concatenate(proportion_class_preference,1)
            
            accuracy_score_benchmark = np.mean(np.array(labeled_data)[train] == np.max(class_labels))
            baseline_proportions = []
            for m in range(num_labels):
                baseline_proportions.append((np.sum(labeled_data==class_labels[m])+0)/(np.sum((labeled_data !=np.max(class_labels)+1)+0)))
            if np.sum(np.isnan(preference_by_class_matrix[:,0])+0) > 0:
                preference_by_class_matrix[np.array(np.isnan(preference_by_class_matrix[:,0]).T)[0],:] = np.repeat(np.matrix(baseline_proportions)/np.sum(baseline_proportions), # note: compute RELATIVE proportions
                                                                                                               np.sum(np.isnan(preference_by_class_matrix[:,0])+0),
                                                                                                            axis=0)
            if len(np.unique(membership_y))>2: ## k>2 classes
                micro_auc.append(metrics.roc_auc_score(label_binarize(membership_y[test],np.unique(membership_y)), ## this piece "binarizes" testing labels
                                                   preference_by_class_matrix[test,:],average='micro'))
                wt_auc.append(metrics.roc_auc_score(label_binarize(membership_y[test],np.unique(membership_y)),
                                                    preference_by_class_matrix[test,:],average='weighted'))
            
                accuracy.append(metrics.accuracy_score(label_binarize(membership_y[test],np.unique(membership_y)), preference_by_class_matrix[test,:]))

            else: ## k=2 classes
                micro_auc.append(metrics.roc_auc_score(label_binarize(membership_y[test],np.unique(membership_y)),
                                  preference_by_class_matrix[test,:][:,1]-preference_by_class_matrix[test,:][:,0],average='micro'))
                wt_auc.append(metrics.roc_auc_score(label_binarize(membership_y[test],np.unique(membership_y)),
                                                                      preference_by_class_matrix[test,:][:,1]-preference_by_class_matrix[test,:][:,0],average='weighted'))
                accuracy.append(metrics.accuracy_score(label_binarize(membership_y[test],np.unique(membership_y)),
                                                       ((preference_by_class_matrix[test,:][:,1]) >accuracy_score_benchmark)+0))
        mean_accuracy.append(np.mean(accuracy)) #placeholder
        se_accuracy.append(np.std(accuracy)) #placeholder
        mean_micro_auc.append(np.mean(micro_auc))
        se_micro_auc.append(np.std(micro_auc))
        mean_wt_auc.append(np.mean(wt_auc))
        se_wt_auc.append(np.std(wt_auc))
    return(mean_accuracy, se_accuracy, mean_micro_auc,se_micro_auc, mean_wt_auc,se_wt_auc)
