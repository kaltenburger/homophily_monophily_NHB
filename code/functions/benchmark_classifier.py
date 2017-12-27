## 4/24/2017
## about: classifier that assigns labels based on class proportions in the training data samples


## generic random classifier
def random_classifier(adj_matrix_input, Membership, num_unlabeled, num_iter, cv_setup = None):
    adj_matrix = np.copy(adj_matrix_input)
    n = len(Membership)
    classes = np.sort(np.unique(Membership))
    class_labels = np.array(range(len(classes)))
    Membership_update = np.copy(Membership)
    for j in range(len(classes)):
        Membership_update[Membership_update == classes[j]] = class_labels[j]
    
    mean_accuracy_initially_labeled = []
    se_accuracy_initially_labeled = []

    for i in range(len(num_unlabeled)):
        print num_unlabeled[i]
        if cv_setup=='stratified':
            k_fold = cross_validation.StratifiedShuffleSplit((Membership), n_iter=num_iter,
                                                             test_size=num_unlabeled[i],
                                                             random_state=0)
        else:
            k_fold = cross_validation.ShuffleSplit(len(Membership), n_iter=num_iter,
                                                   test_size=num_unlabeled[i],
                                                   random_state=0)
        accuracy = []
        for k, (train, test) in enumerate(k_fold):
            proportion_classes = []
            for i in range(len(classes)):
                proportion_classes.append( np.sum((Membership_update[train] ==class_labels[i])+0)/len(train))
            result_score = np.matrix([(proportion_classes)]*len(test))
            accuracy.append(metrics.roc_auc_score(label_binarize(Membership_update[test],np.unique(Membership_update)),
                                                  result_score[:,1] - result_score[:,0],average='weighted'))
        mean_accuracy_initially_labeled.append(np.mean(accuracy))
        se_accuracy_initially_labeled.append(np.std(accuracy))
    return(mean_accuracy_initially_labeled, se_accuracy_initially_labeled)



