from __future__ import division
from sklearn.preprocessing import label_binarize


def LINK(num_unlabeled, membership_y, feature_x, clf, num_iter, cv_setup=None):
    class_labels = np.sort(np.unique(np.array(membership_y))) #unique label IDs
    mean_accuracy = []
    se_accuracy = []
    mean_micro_auc = []
    se_micro_auc = []
    mean_wt_auc = []
    se_wt_auc = []
    for i in range(len(num_unlabeled)):
        print num_unlabeled[i]
        if cv_setup=='stratified':
            k_fold = cross_validation.StratifiedShuffleSplit((membership_y), n_iter=num_iter,
                                               test_size=num_unlabeled[i],
                                               random_state=0)
        else:
            k_fold = cross_validation.ShuffleSplit(len(membership_y), n_iter=num_iter,
                                                         test_size=num_unlabeled[i],
                                                         random_state=0)
        accuracy = []
        micro_auc = []
        wt_auc = []
        for k, (train, test) in enumerate(k_fold):
            #if k==0:
            #print train
            clf.fit(feature_x[train], np.ravel(membership_y[train]))
            pred = clf.predict(feature_x[test])
            prob = clf.predict_proba(feature_x[test])
            #accuracy.append(metrics.accuracy_score(membership_y[test], pred,  normalize = True))
            labeled_data = np.copy(np.array(membership_y))
            ground_truth_testing = np.array(labeled_data)[test]
            labeled_data[test]=np.max(class_labels)+1 # ignore testing labels -- don't have access as part of training -- want to assing test label outside of possible training labels
            
            accuracy_score_benchmark = np.mean(np.array(labeled_data)[train] == np.max(class_labels))

            # auc scores
            if len(np.unique(membership_y))>2:
                micro_auc.append(metrics.roc_auc_score(label_binarize(membership_y[test],np.unique(membership_y)), prob,  average = 'micro'))
                wt_auc.append(metrics.roc_auc_score(label_binarize(membership_y[test],np.unique(membership_y)), prob,
                                                                                                                             average = 'weighted'))
            else:
                micro_auc.append(metrics.roc_auc_score(label_binarize(membership_y[test],np.unique(membership_y)),
                                                                        prob[:,1],average='micro'))
                wt_auc.append(metrics.roc_auc_score(label_binarize(membership_y[test],np.unique(membership_y)),
                                                                            prob[:,1],average='weighted'))

                y_true = label_binarize(membership_y[test],np.unique(membership_y))
                y_pred = np.array(((prob[:,1]) >accuracy_score_benchmark)+0)
                accuracy.append(f1_score(y_true, y_pred, average='macro'))#, pos_label=1) )
                tn, fp, fn, tp = confusion_matrix(label_binarize(membership_y[test],np.unique(membership_y)),
                                                  ((prob[:,1]) >accuracy_score_benchmark)+0).ravel()
                    #accuracy.append((tn/(fp+tn)*0.5 + tp/(tp+fn))*0.5)

        mean_accuracy.append(np.mean(accuracy))
        se_accuracy.append(np.std(accuracy))
        mean_micro_auc.append(np.mean(micro_auc))
        se_micro_auc.append(np.std(micro_auc))
        mean_wt_auc.append(np.mean(wt_auc))
        se_wt_auc.append(np.std(wt_auc))
    return(mean_accuracy, se_accuracy, mean_micro_auc,se_micro_auc, mean_wt_auc,se_wt_auc)
