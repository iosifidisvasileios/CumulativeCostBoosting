# coding: utf-8
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier

__all__ = ['AdaMEC']


class AdaMEC(AdaBoostClassifier):
    """
    Implementation of the cost sensitive variants of AdaBoost; Adacost and AdaC1-3

    Reference: Nikolaou et al Mach Learn (2016) 104:359–384.
    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='AdaMEC',
                 class_weight=None,
                 random_state=None,
                 calibration_method='isotonic',
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
        self.calibration_method = calibration_method
        self.n_estimators = n_estimators

    def fit(self, X, y, sample_weight=None):
        """
        Build a boosted classifier from the training set (X, y).

        :param X:{array-like, sparse matrix}, shape (n_samples, n_features)
               Matrix containing the training data.
        :param y: array-like, shape (n_samples,)
               Corresponding label for each sample in X.
        :param sample_weight: array-like of shape = [n_samples], optional
               Sample weights. If None, the sample weights are initialized to the class weights
        :return: object; Return self
        """

        self.AdaBoostUncal = AdaBoostClassifier(algorithm='SAMME', n_estimators=self.n_estimators)
        self.AdaBoostUncal.fit(X, y)

    def set_costs(self, class_weight):
        # Check parameters
        self.positive_weight = class_weight[1]
        self.negative_weight = class_weight[0]

    def predict(self, X):
        """
        Predict classes for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
               Matrix containing the input samples.
        :return: y_predicted; predicted labels for X
        """
        y_pred = self.AdaBoostUncal.predict_proba(X)

        y_pred_output = np.zeros(X.shape[0])
        y_pred_output[np.where( self.positive_weight * y_pred[:, 1] > self.negative_weight * abs(y_pred[:, 0]))] = 1

        return y_pred_output

    def predict_proba(self, X):
        return self.AdaBoostUncal.predict_proba(X)
