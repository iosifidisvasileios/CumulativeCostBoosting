# coding: utf-8
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier

__all__ = ['CGAda']

from sklearn.utils import check_array, column_or_1d, compute_class_weight
from sklearn.utils.multiclass import check_classification_targets

from Competitors.CostBoostingAlgorithms import CostSensitiveAlgorithms


class CGAda(AdaBoostClassifier):
    """
    Implementation of the cost sensitive variants of AdaBoost; Adacost and AdaC1-3

    Reference: Nikolaou et al Mach Learn (2016) 104:359–384.
    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='CGAda',
                 class_weight=None,
                 random_state=None,
                 debug=False):
        """

        :param base_estimator: object, optional (default=DecisionTreeClassifier)
               The base estimator from which the boosted ensemble is built.
               Support for sample weighting is required, as well as proper 'classes_' and 'n_classes_' attributes.
        :param n_estimators: int, optional (default=50)
               The maximum number of estimators at which boosting is terminated.
               In case of perfect fit, the learning procedure is stopped early.
        :param learning_rate: float, optional (default=1.)
               Learning rate shrinks the contribution of each classifier by "learning_rate".
               There is a trade-off between "learning_rate" and "n_estimators".
        :param algorithm: algorithm: {'adacost', 'adac1', 'adac2', 'adac3'}, optional (default='adacost')
        :param class_weight: dict, list of dicts, “balanced” or None, default=None
               Weights associated with classes in the form {class_label: weight}. The “balanced” mode uses 
               the values of y to automatically adjust weights inversely proportional to class frequencies 
               in the input data as n_samples / (n_classes * np.bincount(y))If not given, all classes are
               supposed to have weight one. For multi-output problems, a list of dicts can be provided in 
               the same order as the columns of y.
        :param random_state: int, RandomState instance or None, optional (default=None)
               If int, random_state is the seed used by the random number generator; If RandomState instance,
               random_state is the random number generator; If None, the random number generator is the RandomState
               instance used by 'np.random'.
        """
        super(AdaBoostClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

        self.algorithm = algorithm
        self.class_weight = class_weight
        self.error = 0
        self.debug = debug
        self.n_estimators = n_estimators

    def fit(self, X, y):
        clf = CostSensitiveAlgorithms(n_estimators=self.n_estimators, algorithm='CGAda', class_weight=self.class_weight)
        clf.fit(X, y)
        self.clf = clf

    def predict(self, X):
        """
        Predict classes for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
               Matrix containing the input samples.
        :return: y_predicted; predicted labels for X
        """
        # scores_CalibratedAdaMEC = self.AdaBoostCal.predict_proba(X)[:, 1]  # Positive Class scores
        y_pred = self.clf.predict(X)

        return y_pred

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

