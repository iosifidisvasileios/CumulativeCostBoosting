import math
from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.base import ClassifierMixin, is_regressor
from sklearn.ensemble import BaseEnsemble
import six
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.extmath import softmax
from sklearn.utils.validation import has_fit_parameter, check_is_fitted, check_array, check_X_y, check_random_state, \
    _check_sample_weight

__all__ = [
    'RareBoost',
]


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
                 debug=False,
                 random_state=None):

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        self.learning_rate = learning_rate
        self.random_state = random_state
        self.debug = debug

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

    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier/regressor from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        Returns
        -------
        self : object
        """
        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        X, y = self._validate_data(X, y)
        sample_weight = _check_sample_weight(sample_weight, X, np.float64)
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative weights")

        # Check parameters
        self._validate_estimator()
        self.classifiers = None
        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_pos = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_weights_neg = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        self._class_weights_pos = []
        self._class_weights_neg = []
        self.training_error = []

        self.smp_w_tmp = []
        self.estimator_tmp = []
        random_state = check_random_state(self.random_state)
        for iboost in range(self.n_estimators):

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            sample_weight /= sample_weight_sum
            self.smp_w_tmp.append(np.copy(sample_weight))

            # Boosting step
            sample_weight, estimator_weight_pos, estimator_weight_neg, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state)

            # Early termination
            if sample_weight is None or np.isnan(sample_weight).any():
                break

            # Stop if error is zero
            if estimator_error == 0:
                break

        return self

    @abstractmethod
    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost.

        Warning: This method needs to be overridden by subclasses.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        sample_weight : array-like of shape (n_samples,)
            The current sample weights.

        random_state : RandomState
            The current random number generator

        Returns
        -------
        sample_weight : array-like of shape (n_samples,) or None
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
        feature_importances_ : ndarray of shape (n_features,)
            The feature importances.
        """
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError("Estimator not fitted, "
                             "call `fit` before `feature_importances_`.")

        self.estimator_weights_ = np.mean([self.estimator_weights_pos[:self.classifiers],  self.estimator_weights_neg[:self.classifiers]], axis=0)


        try:
            norm = self.estimator_weights_.sum()
            return (sum(weight * clf.feature_importances_ for weight, clf
                        in zip(self.estimator_weights_, self.estimators_))
                    / norm)

        except AttributeError:
            raise AttributeError(
                "Unable to compute feature importances "
                "since base_estimator does not have a "
                "feature_importances_ attribute")


class RareBoost(BaseWeightBoosting, ClassifierMixin):
    """An AdaBoost classifier.

    An AdaBoost [1] classifier is a meta-estimator that begins by fitting a
    classifier on the original dataset and then fits additional copies of the
    classifier on the same dataset but where the weights of incorrectly
    classified instances are adjusted such that subsequent classifiers focus
    more on difficult cases.

    This class implements the algorithm known as AdaBoost-SAMME [2].

    Read more in the :ref:`User Guide <adaboost>`.

    .. versionadded:: 0.14

    Parameters
    ----------
    base_estimator : object, optional (default=None)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper
        ``classes_`` and ``n_classes_`` attributes. If ``None``, then
        the base estimator is ``DecisionTreeClassifier(max_depth=1)``.

    n_estimators : int, optional (default=50)
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
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    classes_ : array of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.

    estimator_weights_ : array of floats
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : array of floats
        Classification error for each estimator in the boosted
        ensemble.

    feature_importances_ : ndarray of shape (n_features,)
        The feature importances if supported by the ``base_estimator``.

    See Also
    --------
    AdaBoostRegressor
        An AdaBoost regressor that begins by fitting a regressor on the
        original dataset and then fits additional copies of the regressor
        on the same dataset but where the weights of instances are
        adjusted according to the error of the current prediction.

    GradientBoostingClassifier
        GB builds an additive model in a forward stage-wise fashion. Regression
        trees are fit on the negative gradient of the binomial or multinomial
        deviance loss function. Binary classification is a special case where
        only a single regression tree is induced.

    sklearn.tree.DecisionTreeClassifier
        A non-parametric supervised learning method used for classification.
        Creates a model that predicts the value of a target variable by
        learning simple decision rules inferred from the data features.

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
                 algorithm='SAMME',
                 debug=False,
                 random_state=None):

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            debug=debug,
            random_state=random_state)

        self.debug = debug
        self.algorithm = algorithm

    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, the sample weights are initialized to
            ``1 / n_samples``.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Check that algorithm is supported
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Fit
        return super().fit(X, y, sample_weight)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super()._validate_estimator(
            default=DecisionTreeClassifier(max_depth=1))

        if not has_fit_parameter(self.base_estimator_, "sample_weight"):
            raise ValueError("%s doesn't support sample_weight."
                             % self.base_estimator_.__class__.__name__)

    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME discrete algorithm."""

        estimator = self._make_estimator(random_state=random_state)
        estimator.fit(X, y, sample_weight=sample_weight)
        self.estimator_tmp.append(estimator)

        y_predict = estimator.predict(X)

        tp = 0.
        tn = 0.
        fn = 0.
        fp = 0.
        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        estimator_error = None

        for idx, row in enumerate(sample_weight):
            if y[idx] == 1 and y_predict[idx] != 1:
                fn += sample_weight[idx]
            elif y[idx] != 1 and y_predict[idx] == 1:
                fp += sample_weight[idx]
            elif y[idx] == 1 and y_predict[idx] == 1:
                tp += sample_weight[idx]
            elif y[idx] != 1 and y_predict[idx] != 1:
                tn += sample_weight[idx]

        if tp < fp:
            raise ValueError('violation: tp < fp   ! ')
            return None, None, None

        if tn < fn:
            raise ValueError('violation: tn < fn   ! ')
            return None, None, None

        if fn <= 0 and fp <= 0:
            return sample_weight, 1., 1, 0.

        # this implementation is based on https://github.com/EdwinTh/rareboost
        try:
            estimator_weight_pos = 0.5 * np.log((tp) / fp)
        except:
            estimator_weight_pos = float('nan')

        try:
            estimator_weight_neg = 0.5 * np.log((tn) / fn)
        except:
            estimator_weight_neg = float('nan')

        if math.isnan(estimator_weight_pos) or math.isinf(estimator_weight_pos):
            estimator_weight_pos = 0.5 * np.log(len(y))
        if math.isnan(estimator_weight_neg) or math.isinf(estimator_weight_neg):
            estimator_weight_neg = 0.5 * np.log(len(y))

        self.estimator_weights_pos[iboost] = estimator_weight_pos
        self.estimator_weights_neg[iboost] = estimator_weight_neg
        self.estimator_errors_[iboost] = estimator_error

        if self.debug:
            self.training_error.append(1 - balanced_accuracy_score(y, self.predict(X)))
            self._class_weights_pos.append(sample_weight[y == 1].sum())
            self._class_weights_neg.append(sample_weight[y != 1].sum())

        # Only boost the weights if I will fit again
        if not iboost == self.n_estimators - 1:
            for idx, row in enumerate(sample_weight):
                if y[idx] != 1 and y_predict[idx] == 1:
                    sample_weight[idx] *= np.exp(estimator_weight_pos)
                elif y[idx] == 1 and y_predict[idx] == 1:
                    sample_weight[idx] *= np.exp(-estimator_weight_pos)
                elif y[idx] == 1 and y_predict[idx] != 1:
                    sample_weight[idx] *= np.exp(estimator_weight_neg)
                elif y[idx] != 1 and y_predict[idx] != 1:
                    sample_weight[idx] *= np.exp(-estimator_weight_neg)

        return sample_weight, estimator_weight_pos, estimator_weight_neg, estimator_error

    def get_confidence_scores(self, X):
        check_is_fitted(self)
        X = self._validate_data(X)
        classes = self.classes_[:, np.newaxis]
        neg_idx = np.where(np.hstack(classes) != 1)[0][0]
        pos_idx = np.where(np.hstack(classes) == 1)[0][0]

        pred_pos = sum((estimator.predict(X) == classes).T[:, pos_idx] * w1
                       for estimator, w1 in zip(self.estimators_,
                                                self.estimator_weights_pos))

        pred_neg = sum((estimator.predict(X) == classes).T[:, neg_idx] * w1
                       for estimator, w1 in zip(self.estimators_,
                                                self.estimator_weights_neg))

        pred = np.column_stack((pred_neg, pred_pos))

        pred /= (self.estimator_weights_pos.sum() + self.estimator_weights_neg.sum())
        pred[:, neg_idx] *= -1
        return pred.sum(axis=1)

    def predict(self, X):
        """Predict classes for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        X = self._validate_data(X)

        pred = self.decision_function(X)

        # if self.n_classes_ == 2:
        #     return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def set_classifiers(self, classifiers):
        self.classifiers = classifiers

    def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        score : array, shape = [n_samples, k]
            The decision function of the input samples. The order of
            outputs is the same of that of the :term:`classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self)
        X = self._validate_data(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        neg_idx = np.where(np.hstack(classes) != 1)[0][0]
        pos_idx = np.where(np.hstack(classes) == 1)[0][0]

        if self.classifiers:
            estimators = self.estimators_[:self.classifiers]
            alphas_pos = self.estimator_weights_pos[:self.classifiers]
            alphas_neg = self.estimator_weights_neg[:self.classifiers]
        else:
            estimators = self.estimators_
            alphas_pos = self.estimator_weights_pos
            alphas_neg = self.estimator_weights_neg


        pred_pos = sum((estimator.predict(X) == classes).T[:, pos_idx] * w1
                       for estimator, w1 in zip(estimators,
                                                alphas_pos))

        pred_neg = sum((estimator.predict(X) == classes).T[:, neg_idx] * w1
                       for estimator, w1 in zip(estimators,
                                                alphas_neg))

        pred = np.column_stack((pred_neg, pred_pos))

        pred /= (alphas_pos.sum() + alphas_neg.sum())
        # if n_classes == 2:
        #     pred[:, neg_idx] *= -1
        #     return pred.sum(axis=1)
        return pred


    @staticmethod
    def _compute_proba_from_decision(decision, n_classes):
        """Compute probabilities from the decision function.

        This is based eq. (4) of [1] where:
            p(y=c|X) = exp((1 / K-1) f_c(X)) / sum_k(exp((1 / K-1) f_k(X)))
                     = softmax((1 / K-1) * f(X))

        References
        ----------
        .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost",
               2009.
        """
        if n_classes == 2:
            decision = np.vstack([-decision, decision]).T / 2
        else:
            decision /= (n_classes - 1)
        return softmax(decision, copy=False)


    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities of the classifiers
        in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        p : array of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of
            outputs is the same of that of the :term:`classes_` attribute.
        """

        n_classes = self.n_classes_

        return self.decision_function(X)

