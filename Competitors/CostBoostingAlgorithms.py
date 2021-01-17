# coding: utf-8
import numpy as np
from sklearn.base import is_regressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaseEnsemble
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from sklearn.tree import BaseDecisionTree, DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import (check_random_state,
                           check_X_y,
                           check_array,
                           compute_class_weight,
                           column_or_1d)
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.metrics import balanced_accuracy_score

__all__ = ['CostSensitiveAlgorithms']


class CostSensitiveAlgorithms(AdaBoostClassifier):
    """
    Implementation of the cost sensitive variants of AdaBoost; Adacost and AdaC1-3

    Reference: Nikolaou et al Mach Learn (2016) 104:359–384.
    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm=None,
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

    def fit(self, X, y ):
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
        # Check parameters
        # if self.learning_rate <= 0:
        #     raise ValueError("learning_rate must be greater than zero")
        #
        # if (self.base_estimator is None or
        #         isinstance(self.base_estimator, (BaseDecisionTree,
        #                                          BaseEnsemble))):
        #     DTYPE = np.float64
        #     dtype = DTYPE
        #     accept_sparse = 'csc'
        # else:
        #     dtype = None
        #     accept_sparse = ['csr', 'csc']
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        # X, y = self._validate_data(X, y)
        # self._validate_targets(y)
        # sample_weight = _check_sample_weight(sample_weight, X, np.float64)
        # sample_weight /= sample_weight.sum()

        y = self._validate_targets(y)

        sample_weight = np.copy(y).astype(float)
        for n in range(len(self.classes)):
            sample_weight[y == n] = self.class_weight_[n]

        # Check that the sample weights sum is positive
        if sample_weight.sum() <= 0:
            raise ValueError(
                "Attempting to fit with a non-positive weighted number of samples.")
        sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)




        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative weights")

        if self.algorithm is None:
            self.algorithm = "AdaCost"

        self.classifiers = None
        # Check parameters
        self._validate_estimator()

        # Clear any previous fit results
        self.estimators_ = []

        self.training_error = []
        self._class_weights_pos = []
        self._class_weights_neg = []
        self.estimator_alphas_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)
        if self.debug:
            self.predictions_array = np.zeros([X.shape[0], 2])

        random_state = check_random_state(1)

        for iboost in range(self.n_estimators):
            if self.debug:
                self._class_weights_pos.append(sample_weight[y == 1].sum())
                self._class_weights_neg.append(sample_weight[y != 1].sum())


            # Boosting step
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state)
            # print('mu class debug', sample_weight, estimator_weight, estimator_error)
            if self.error == 1:
                break

            # Early termination
            if sample_weight is None:
                break
            self.estimator_alphas_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            if self.debug:
                self.predictions_array += (self.estimators_[iboost].predict(X) == self.classes_[:, np.newaxis]).T * self.estimator_alphas_[iboost]
                tn, fp, fn, tp = confusion_matrix(y,
                                                  self.classes_.take(np.argmax(self.predictions_array, axis=1), axis=0),
                                                  labels=[0, 1]).ravel()

                self.training_error.append(1 - ((float(tp)) / (tp + fn) + (float(tn))/(tn + fp))/2.)
            # Stop if error is zero
            if estimator_error == 0:
                break

        return self

    def _validate_data(self, X, y=None):

        # Accept or convert to these sparse matrix formats so we can
        # use _safe_indexing
        accept_sparse = ['csr', 'csc']
        if y is None:
            ret = check_array(X,
                              accept_sparse=accept_sparse,
                              ensure_2d=False,
                              allow_nd=True,
                              dtype=None)
        else:
            ret = check_X_y(X, y,
                            accept_sparse=accept_sparse,
                            ensure_2d=False,
                            allow_nd=True,
                            dtype=None,
                            y_numeric=is_regressor(self))
        return ret
    def _boost(self, iboost, X, y, sample_weight, random_state):
        """
        Implement a single boost.

        Perform a single boost according to the algorithm selected and return the updated
        sample weights.

        :param iboost: int
               The index of the current boost iteration
        :param X:{array-like, sparse matrix}, shape (n_samples, n_features)
               Matrix containing the training data.
        :param y: array-like, shape (n_samples,)
               Corresponding label for each sample in X.
        :param sample_weight: array-like of shape = [n_samples], optional
               Sample weights. If None, the sample weights are initialized to the class weights
        :param random_state: int, RandomState instance or None, optional (default=None)
               If int, random_state is the seed used by the random number generator; If RandomState instance,
               random_state is the random number generator; If None, the random number generator is the RandomState
               instance used by 'np.random'.
        :return: sample_weight {array-like of shape = [n_samples]}, estimator_weight {float}, estimator_error {float}
                Returns updates values for sample weights, estimator weight and estimator error
        """
        # if len(np.argwhere(np.isnan(sample_weight))) != 0:
        #     self.error = 1
        #     return None, None, None
        estimator = self._make_estimator(random_state=1)
        estimator.fit(X, y, sample_weight=sample_weight)
        y_predict = estimator.predict(X)
        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

            # assign class weight to each sample index
            costs = np.copy(y).astype(float)
            if self.algorithm == "AdaCost":
                for n in range(self.n_classes_):
                    costs[y != n] = self.class_weight_[n]
            else:
                for n in range(self.n_classes_):
                    costs[y == n] = self.class_weight_[n]
            self.cost_ = costs

        # Instances incorrectly classified
        incorrect = y_predict != y

        if self.algorithm == "AdaBoost" or self.algorithm == "AdaCost" or self.algorithm == "CSB1" \
                or self.algorithm == "CSB2" or self.algorithm == "CGAda":
            estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        elif self.algorithm in ['AdaC1', 'AdaC2']:
            estimator_error = np.mean(np.average(incorrect, weights=sample_weight * self.cost_, axis=0))
        elif self.algorithm == "AdaC3":
            estimator_error = np.mean(np.average(incorrect, weights=sample_weight * np.power(self.cost_, 2), axis=0))
        else:
            raise ValueError("Algorithms 'adacost', 'CGAda', 'adac1', 'adac2' and 'adac3' are accepted;" \
                             " got {0}".format(self.algorithm))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_

        # Stop if the error is at least as bad as random guessing
        if estimator_error >= 1. - (1. / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in AdaBoostClassifier '
                                 'ensemble is worse than random, ensemble '
                                 'can not be fit.', self.algorithm, self.class_weight_)
            return None, None, None

        if self.algorithm == "AdaBoost" or self.algorithm == "AdaCost" or self.algorithm == "AdaC2" or \
                self.algorithm == "CSB1" or self.algorithm == "CSB2"  or self.algorithm == "CGAda":
            # estimator_weight = self.learning_rate * 0.5 * (
            #     np.log((1. - estimator_error) / estimator_error))
            estimator_weight = self.learning_rate * (
                    np.log((1. - estimator_error) / estimator_error) +
                    np.log(n_classes - 1.))

        elif self.algorithm == "AdaC1":
            estimator_weight = self.learning_rate * 0.5 * (
                np.log((1 + (1. - estimator_error) - estimator_error) /
                       (1 - (1. - estimator_error) + estimator_error)))

        elif self.algorithm == "AdaC3":
            c2d_truly_classified = np.sum(np.power(self.cost_[~ incorrect], 2) * sample_weight[~ incorrect])
            c2d_misclassified = np.sum(np.power(self.cost_[incorrect], 2) * sample_weight[incorrect])
            estimator_weight = self.learning_rate * 0.5 * (
                np.log(
                    (np.sum(sample_weight * self.cost_) + c2d_truly_classified - c2d_misclassified) /
                    (np.sum(sample_weight * self.cost_) - c2d_truly_classified + c2d_misclassified))
            )

        # Only boost the weights if it will fit again
        if iboost < self.n_estimators - 1:
            if self.algorithm == "AdaBoost" or self.algorithm == "CGAda":
                nominator = sample_weight * np.exp(estimator_weight * incorrect * (sample_weight > 0))
                sample_weight = nominator / np.sum(nominator)
                # sample_weight *=            np.exp(estimator_weight * incorrect * (sample_weight > 0))


            elif self.algorithm == "AdaCost":
                beta = np.copy(self.cost_).astype(float)
                beta[y == y_predict] = np.array(list(map(lambda x: 0.5 * x + 0.5, self.cost_[y == y_predict])))
                beta[y != y_predict] = np.array(list(map(lambda x: -0.5 * x + 0.5, self.cost_[y != y_predict])))
                # Only boost positive weights
                nominator = sample_weight * np.exp(estimator_weight * incorrect * beta * (sample_weight > 0))
                sample_weight = nominator / np.sum(nominator)
            elif self.algorithm == "AdaC1":
                nominator = sample_weight * np.exp(self.cost_ * estimator_weight * incorrect * (sample_weight > 0))
                sample_weight = nominator / np.sum(nominator)
            elif self.algorithm == "AdaC2":
                nominator = sample_weight * self.cost_ * np.exp(estimator_weight * incorrect * (sample_weight > 0))
                sample_weight = nominator / np.sum(nominator)
            elif self.algorithm == "AdaC3":
                nominator = sample_weight * self.cost_ * np.exp(
                    estimator_weight * self.cost_ * incorrect * (sample_weight > 0))
                sample_weight = nominator / np.sum(nominator)
            elif self.algorithm == "CSB1":
                nominator = sample_weight * np.exp(self.cost_ * incorrect * (sample_weight > 0))
                sample_weight = nominator / np.sum(nominator)
            elif self.algorithm == "CSB2":
                nominator = sample_weight * self.cost_ * np.exp(estimator_weight * incorrect * (sample_weight > 0))
                sample_weight = nominator / np.sum(nominator)
                # sample_weight *= self.cost_ * np.exp(estimator_weight * incorrect * (sample_weight > 0))
            else:
                raise ValueError("algorithm %s is not supported" % self.algorithm)

        return sample_weight, estimator_weight, estimator_error
    #
    # def _validate_estimator(self):
    #     """
    #     Check the estimator and set the base_estimator_ attribute.
    #     """
    #     super(AdaBoostClassifier, self)._validate_estimator(
    #         default=DecisionTreeClassifier(max_depth=1, class_weight=self.class_weight_))

    def _validate_targets(self, y):
        """
        Validation of y and class_weight.

        :param y: array-like, shape (n_samples,)
               Corresponding label for each sample in X.
        :return: validated y {array-like, shape (n_samples,)}
        """
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        cls, y = np.unique(y_, return_inverse=True)
        class_weight_ = compute_class_weight(self.class_weight, cls, y_)

        self.class_weight_ = {i: class_weight_[i] for i in range(len(class_weight_))}
        if len(cls) < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d"
                % len(cls))

        self.classes = cls
        return np.asarray(y, dtype=np.float64, order='C')

    def predict(self, X):
        """
        Predict classes for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
               Matrix containing the input samples.
        :return: y_predicted; predicted labels for X
        """
        pred = self.decision_function(X)
        # >>> removed binary special case
        # if self.n_classes_ == 2:
        #    return self.classes_.take(pred == 0, axis=0)
        # <<<

        return self.classes.take(np.argmax(pred, axis=1), axis=0)

    def set_classifiers(self, classifiers):
        self.classifiers = classifiers

    def decision_function(self, X):
        """
        Compute the decision function of X
        :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
               Matrix containing the input samples.
        :return: score : array, shape = [n_samples, k]
                 The decision function of the input samples. The order of outputs is the same of
                 that of the 'classes_' attribute.
        """
        check_is_fitted(self, "n_classes_")
        # X = super()._validate_X_predict(X)

        if self.classifiers:
            estimators = self.estimators_[:self.classifiers]
            alphas = self.estimator_alphas_[:self.classifiers]
        else:
            estimators = self.estimators_
            alphas = self.estimator_alphas_

        classes = self.classes_[:, np.newaxis]
        pred = sum((estimator.predict(X) == classes).T * w
                   for estimator, w in zip(estimators,
                                           alphas))

        pred /= alphas.sum()
        # pred[:, 0] *= -1
        # return pred.sum(axis=1)
        # >>> removed binary special case
        # if n_classes == 2:
        #    pred[:, 0] *= -1
        #    return pred.sum(axis=1)
        # <<<
        return pred

    def get_confidence_scores(self, X):

        classes = self.classes_[:, np.newaxis]
        pred = sum((estimator.predict(X) == classes).T * w
                   for estimator, w in zip(self.estimators_,
                                           self.estimator_alphas_))

        pred /= self.estimator_alphas_.sum()
        pred[:, 0] *= -1
        return pred.sum(axis=1)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities of the classifiers
        in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        p : array of shape = [n_samples]
            The class probabilities of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
        """
        check_is_fitted(self, "n_classes_")

        n_classes = self.n_classes_
        # X = self._validate_X_predict(X)

        if n_classes == 1:
            return np.ones((X.shape[0], 1))

        proba = sum(estimator.predict_proba(X) * w for estimator, w in zip(self.estimators_, self.estimator_alphas_))

        proba /= self.estimator_alphas_.sum()
        proba = np.exp((1. / (n_classes - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        return proba