"""Weight Boosting

This module contains weight boosting estimators for both classification and
regression.

The module structure is the following:

- The ``BaseWeightBoosting`` base class implements a common ``fit`` method
  for all the estimators in the module. Regression and classification
  only differ from each other in the loss function that is optimized.

- ``AdaCostClassifier`` implements adaptive boosting (AdaBoost-SAMME) for
  classification problems.

- ``AdaBoostRegressor`` implements adaptive boosting (AdaBoost.R2) for
  regression problems.
"""

# Authors: Noel Dawe <noel@dawe.me>
#          Gilles Louppe <g.louppe@gmail.com>
#          Hamzeh Alsalhi <ha258@cornell.edu>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#
# License: BSD 3 clause
from abc import ABCMeta, abstractmethod
from collections import defaultdict

import numpy as np
import six
from sklearn.base import  ClassifierMixin, is_regressor
from sklearn.ensemble import BaseEnsemble
# from sklearn.externals import six
from sklearn.tree import BaseDecisionTree, DecisionTreeClassifier
from sklearn.utils.validation import has_fit_parameter, check_is_fitted, check_array, check_X_y, check_random_state

__all__ = [
    'AdaCC',
]

from sklearn.metrics import confusion_matrix

class BaseWeightBoosting(six.with_metaclass(ABCMeta, BaseEnsemble)):
    """Base class for AdaBoost estimators.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 estimator_params=tuple(),
                 learning_rate=1.,
                 attributes=None,
                 random_state=None):

        super(BaseWeightBoosting, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        self.attributes = attributes
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier/regressor from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR. The dtype is
            forced to DTYPE from tree._tree if the base classifier of this
            ensemble weighted boosting classifier is a tree or forest.

        y : array-like of shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree))):
            dtype = np.float64
            accept_sparse = 'csc'
        else:
            dtype = np.float64
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype, y_numeric=is_regressor(self))

        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        # Check parameters
        self._validate_estimator()


        self.cost_pos = []
        self.cost_neg = []
        self.training_error = []
        self._class_weights_pos = []
        self._class_weights_neg = []

        self._tpr = []
        self._tnr = []
        self._standard_error = []


        self.classifiers = None
        self.estimators_ = []
        self.estimator_alphas_ = np.zeros(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)

        self.predictions_array = np.zeros([ X.shape[0], 2])

        self.smp_w_tmp = []
        self.estimator_tmp = []
        self.alpha_tmp = []
        for iboost in range(self.n_estimators):

            sample_weight_sum = np.sum(sample_weight)
            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break
            sample_weight /= sample_weight_sum
            self.smp_w_tmp.append(np.copy(sample_weight))

            # Boosting step
            sample_weight, alpha, error = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state)

            # Early termination
            if sample_weight is None:
                break

            # Stop if error is zero
            if error == 0:
                break

        if self.attributes:
            print(self.algorithm, self.employed_attributes)

        return self

    def get_confidence_scores(self, X):
        classes = self.classes_[:, np.newaxis]
        pred = sum(
            (estimator.predict(X) == classes).T * w for estimator, w in zip(self.estimators_, self.estimator_alphas_))
        pred /= self.estimator_alphas_.sum()
        pred[:, 0] *= -1
        return pred.sum(axis=1)

    @abstractmethod
    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost.

        Warning: This method needs to be overridden by subclasses.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape = [n_samples]
            The target values (class labels).

        sample_weight : array-like of shape = [n_samples]
            The current sample weights.

        random_state : numpy.RandomState
            The current random number generator

        Returns
        -------
        sample_weight : array-like of shape = [n_samples] or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        error : float
            The classification error for the current boost.
            If None then boosting has terminated early.
        """
        pass

    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError("Estimator not fitted, "
                             "call `fit` before `feature_importances_`.")

        try:
            norm = self.estimator_alphas_.sum()
            return (sum(weight * clf.feature_importances_ for weight, clf
                        in zip(self.estimator_alphas_, self.estimators_))
                    / norm)

        except AttributeError:
            raise AttributeError(
                "Unable to compute feature importances "
                "since base_estimator does not have a "
                "feature_importances_ attribute")

    def _validate_X_predict(self, X):
        """Ensure that X is in the proper format"""
        if (self.base_estimator is None or
                isinstance(self.base_estimator,
                           (BaseDecisionTree))):
            X = check_array(X, accept_sparse='csr', dtype=np.float64)

        else:
            X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])

        return X


class AdaCC(BaseWeightBoosting, ClassifierMixin):
    """An AdaBoost classifier.

    An AdaBoost [1] classifier is a meta-estimator that begins by fitting a
    classifier on the original dataset and then fits additional copies of the
    classifier on the same dataset but where the weights of incorrectly
    classified instances are adjusted such that subsequent classifiers focus
    more on difficult cases.

    This class implements the algorithm known as AdaBoost-SAMME [2].

    Read more in the :ref:`User Guide <adaboost>`.

    Parameters
    ----------
    base_estimator : object, optional (default=DecisionTreeClassifier)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper `classes_`
        and `n_classes_` attributes.

    n_estimators : integer, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.

    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each classifier by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.

    algorithm : {'SAMME', 'SAMME.R'}, optional (default='SAMME.R')
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities.
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    classes_ : array of shape = [n_classes]
        The classes labels.

    n_classes_ : int
        The number of classes.

    estimator_weights_ : array of floats
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : array of floats
        Classification error for each estimator in the boosted
        ensemble.

    feature_importances_ : array of shape = [n_features]
        The feature importances if supported by the ``base_estimator``.

    See also
    --------
    AdaBoostRegressor, GradientBoostingClassifier, DecisionTreeClassifier

    References
    ----------
    .. [1] Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of
           on-Line Learning and an Application to Boosting", 1995.

    .. [2] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='AdaCC1',
                 random_state=None,
                 attributes=None,
                 amortised=True,
                 debug=False):

        super(AdaCC, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            attributes=attributes,
            random_state=random_state)

        self.debug = debug
        self.algorithm = algorithm
        self.attributes = attributes
        self.employed_attributes = defaultdict(int)
        self.amortised = amortised

    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        y : array-like of shape = [n_samples]
            The target values (class labels).

        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            ``1 / n_samples``.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that algorithm is supported
        if self.algorithm not in ('AdaCC1', 'AdaCC2'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Fit
        return super(AdaCC, self).fit(X, y, sample_weight)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super(AdaCC, self)._validate_estimator(default=DecisionTreeClassifier(max_depth=1, criterion='entropy'))

        #  SAMME-R requires predict_proba-enabled base estimators
        if not has_fit_parameter(self.base_estimator_, "sample_weight"):
            raise ValueError("%s doesn't support sample_weight."
                             % self.base_estimator_.__class__.__name__)

    def calculate_cost(self, labels, predictions):

        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        self.cost_positive = 2 - tp / (tp + fn)
        self.cost_negative = 2 - tn / (tn + fp)

        if self.cost_positive < self.cost_negative:
            self.cost_positive = 1.
        else:
            self.cost_negative = 1.

    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME discrete algorithm."""
        estimator = self._make_estimator(random_state=random_state)
        estimator.fit(X, y, sample_weight=sample_weight)
        y_predict = estimator.predict(X)



        # index = list(estimator.tree_.compute_feature_importances()).index(1.)
        # if self.attributes is not None:
        #     self.employed_attributes[self.attributes[index].split("?")[0]] += 1

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))
        # print(iboost, estimator_error)
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
                                 'can not be fit.')
            return None, None, None

        alpha = self.learning_rate * (np.log((1. - estimator_error) / estimator_error) + np.log(n_classes - 1.))
        self.estimator_alphas_[iboost] = alpha

        if self.amortised:
            self.predictions_array += (estimator.predict(X) == self.classes_[:, np.newaxis]).T * alpha
            self.calculate_cost(y, self.classes_.take(np.argmax(self.predictions_array, axis=1), axis=0))
            # this operation is too slow. replaced it with the above computations
            # self.calculate_cost(y, self.predict(X))
        else:
            self.predictions_array += (estimator.predict(X) == self.classes_[:, np.newaxis]).T * alpha

            self.calculate_cost(y, y_predict)

        # self.smp_w_tmp.append(sample_weight)
        self.estimator_tmp.append(estimator)
        self.alpha_tmp.append(alpha)


        if self.debug:
            # self.training_error.append(1 - accuracy_score(y, self.predict(X)))
            self.cost_pos.append(self.cost_positive)
            self.cost_neg.append(self.cost_negative)
            self._class_weights_pos.append(sample_weight[y == 1].sum())
            self._class_weights_neg.append(sample_weight[y != 1].sum())

            tn, fp, fn, tp  =confusion_matrix(y, self.classes_.take(np.argmax(self.predictions_array, axis=1), axis=0), labels=[0, 1] ).ravel()
            self._tpr.append((float(tp))/(tp + fn))
            self._tnr.append((float(tn))/(tn + fp))
            self.training_error.append(1 - ((float(tp)) / (tp + fn) + (float(tn)) / (tn + fp)) / 2.)

        if not iboost == self.n_estimators - 1:

            if self.algorithm == 'AdaCC1':
                for idx, row in enumerate(sample_weight):
                    if y[idx] == 1 and y_predict[idx] != 1:
                        sample_weight[idx] *= np.exp(self.cost_positive * alpha)
                    elif y[idx] != 1 and y_predict[idx] == 1:
                        sample_weight[idx] *= np.exp(self.cost_negative * alpha)

            elif self.algorithm == 'AdaCC2':
                for idx, row in enumerate(sample_weight):
                    if y[idx] == 1 and y_predict[idx] != 1:
                        sample_weight[idx] *= np.exp(alpha) * self.cost_positive
                    elif y[idx] != 1 and y_predict[idx] == 1:
                        sample_weight[idx] *= np.exp(alpha) * self.cost_negative

        return sample_weight, alpha, estimator_error

    def get_performance_over_iterations(self):
        return self.performance

    def set_classifiers(self, classifiers):
        self.classifiers = classifiers

    def predict(self, X):
        """Predict classes for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        pred = self.decision_function(X)
        #
        # if self.n_classes_ == 2:
        #     return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def decision_function(self, X):
        """Compute the decision function of ``X``.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.
        Returns
        -------
        score : array, shape = [n_samples, k]
            The decision function of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self, "n_classes_")
        X = self._validate_X_predict(X)
        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]

        if self.classifiers:
            estimators = self.estimators_[:self.classifiers]
            alphas = self.estimator_alphas_[:self.classifiers]
        else:
            estimators = self.estimators_
            alphas = self.estimator_alphas_

        pred = sum(
            (estimator.predict(X) == classes).T * w for estimator, w in zip(estimators, alphas))
        pred /= alphas.sum()
        # if n_classes == 2:
        #     pred[:, 0] *= -1
        #     return pred.sum(axis=1)
        return pred

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
        X = self._validate_X_predict(X)

        if n_classes == 1:
            return np.ones((X.shape[0], 1))

        proba = sum(estimator.predict_proba(X) * w for estimator, w in zip(self.estimators_, self.estimator_alphas_))

        proba /= self.estimator_alphas_.sum()
        proba = np.exp((1. / (n_classes - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        return proba

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the weighted mean predicted class log-probabilities of the classifiers
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
        return np.log(self.predict_proba(X))
