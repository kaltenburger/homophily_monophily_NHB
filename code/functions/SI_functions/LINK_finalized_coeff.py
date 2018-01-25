from __future__ import division
from sklearn.preprocessing import label_binarize


def LINK_coeff(pct_unlabeled, membership_y, feature_x, clf, num_iter, cv_setup=None):

    for i in range(len(pct_unlabeled)):
        print(pct_unlabeled[i])
        if cv_setup=='stratified':
            k_fold = cross_validation.StratifiedShuffleSplit((membership_y), n_iter=num_iter,
                                               test_size=pct_unlabeled[i],
                                               random_state=0)
        else:
            k_fold = cross_validation.ShuffleSplit(len(membership_y), n_iter=num_iter,
                                                         test_size=pct_unlabeled[i],
                                                         random_state=0)

        for k, (train, test) in enumerate(k_fold):

            clf.fit(feature_x[train], np.ravel(membership_y[train]))
            coeff = clf.coef_
        return(np.array(coeff)[0])
