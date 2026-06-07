"""Density-based classifiers built from unary density estimators."""
from typing import Any
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from ._forest import UnaryDensityForest


class UnaryDensityBayesClassifier(BaseEstimator, ClassifierMixin):
    """Binary Bayes classifier built from class-conditional densities.

    The classifier fits one density estimator per class and predicts by the
    log-density ratio with an optional empirical prior correction. Input data
    are scaled into a unit hypercube inside ``fit``; this makes the wrapper safe
    to use in cross-validation pipelines without leaking validation data into
    preprocessing.

    Args:
        density_estimator: Estimator with ``fit(X, y=None)`` and either
            ``score_samples(X)`` returning densities or ``score_log_samples(X)``
            returning log densities. If ``None``, a default
            ``UnaryDensityForest`` is used.
        margin: Empty border kept around training data after scaling.
        use_class_prior: Whether to add empirical log-prior odds.
        clip: Whether to clip transformed validation points to ``[0, 1]``.
        eps: Numerical floor for densities and probabilities.
    """

    def __init__(
        self,
        density_estimator: Any | None = None,
        margin: float = 0.03,
        use_class_prior: bool = True,
        clip: bool = True,
        eps: float = 1e-12,
    ) -> None:
        self.density_estimator = density_estimator
        self.margin = margin
        self.use_class_prior = use_class_prior
        self.clip = clip
        self.eps = eps

    def fit(self, X: np.ndarray, y: np.ndarray) -> "UnaryDensityBayesClassifier":
        """Fit class-conditional density estimators.

        Args:
            X: Training samples with shape ``(n_samples, n_features)``.
            y: Binary class labels.

        Returns:
            Fitted classifier.
        """
        self._validate_params()
        X_checked, y_checked = check_X_y(X, y, dtype=np.float64, ensure_2d=True)
        classes, counts = np.unique(y_checked, return_counts=True)
        if classes.shape[0] != 2:
            raise ValueError("UnaryDensityBayesClassifier supports binary labels only.")

        self.classes_ = classes
        self.n_features_in_ = X_checked.shape[1]
        self.class_count_ = counts.astype(np.int64)
        self.class_prior_ = counts.astype(np.float64) / float(X_checked.shape[0])

        X_scaled = self._fit_scaler(X_checked)
        unit_bounds = np.vstack(
            [
                np.zeros(self.n_features_in_, dtype=np.float64),
                np.ones(self.n_features_in_, dtype=np.float64),
            ]
        )

        estimators = []
        for class_label in self.classes_:
            estimator = clone(self._make_density_estimator())
            if hasattr(estimator, "set_params"):
                params = estimator.get_params(deep=False)
                updates = {}
                if "bounds" in params:
                    updates["bounds"] = unit_bounds
                if "bounds_margin" in params:
                    updates["bounds_margin"] = 0.0
                if "clip" in params:
                    updates["clip"] = True
                if updates:
                    estimator.set_params(**updates)
            estimator.fit(X_scaled[y_checked == class_label])
            estimators.append(estimator)

        self.estimators_ = estimators
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return binary log-odds from class-conditional densities.

        Args:
            X: Samples with shape ``(n_samples, n_features)``.

        Returns:
            Log-odds for ``classes_[1]`` against ``classes_[0]``.
        """
        check_is_fitted(self, "estimators_")
        X_scaled = self._transform(X)
        log_density_0 = self._score_log_density(self.estimators_[0], X_scaled)
        log_density_1 = self._score_log_density(self.estimators_[1], X_scaled)
        log_odds = log_density_1 - log_density_0
        if self.use_class_prior:
            prior_0 = max(float(self.class_prior_[0]), self.eps)
            prior_1 = max(float(self.class_prior_[1]), self.eps)
            log_odds += np.log(prior_1) - np.log(prior_0)
        return log_odds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict posterior class probabilities.

        Args:
            X: Samples with shape ``(n_samples, n_features)``.

        Returns:
            Probability matrix with shape ``(n_samples, 2)``.
        """
        log_odds = self.decision_function(X)
        probability_1 = self._sigmoid(log_odds)
        probability_1 = np.clip(probability_1, self.eps, 1.0 - self.eps)
        return np.column_stack([1.0 - probability_1, probability_1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Samples with shape ``(n_samples, n_features)``.

        Returns:
            Predicted labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def _validate_params(self) -> None:
        if self.margin < 0.0 or self.margin >= 0.5:
            raise ValueError("margin must be in [0, 0.5).")
        if self.eps <= 0.0 or self.eps >= 0.5:
            raise ValueError("eps must be in (0, 0.5).")

    def _make_density_estimator(self) -> Any:
        if self.density_estimator is not None:
            return self.density_estimator
        return UnaryDensityForest(random_state=0)

    def _fit_scaler(self, X: np.ndarray) -> np.ndarray:
        lower = np.min(X, axis=0)
        upper = np.max(X, axis=0)
        scale = upper - lower
        zero_width = scale <= 0.0
        if np.any(zero_width):
            scale[zero_width] = 1.0
        self.scale_min_ = lower
        self.scale_width_ = scale
        return self._scale_array(X)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X_checked = check_array(X, dtype=np.float64, ensure_2d=True)
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_checked.shape[1]} features, expected {self.n_features_in_}."
            )
        return self._scale_array(X_checked)

    def _scale_array(self, X: np.ndarray) -> np.ndarray:
        scaled = (X - self.scale_min_) / self.scale_width_
        scaled = self.margin + (1.0 - 2.0 * self.margin) * scaled
        if self.clip:
            scaled = np.clip(scaled, 0.0, 1.0)
        elif np.any((scaled < 0.0) | (scaled > 1.0)):
            raise ValueError("X contains values outside the fitted scaling range.")
        return np.ascontiguousarray(scaled, dtype=np.float64)

    def _score_log_density(self, estimator: Any, X: np.ndarray) -> np.ndarray:
        if hasattr(estimator, "score_log_samples"):
            return np.asarray(estimator.score_log_samples(X), dtype=np.float64)
        density = np.maximum(estimator.score_samples(X), self.eps)
        return np.log(density)

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        out = np.empty_like(values, dtype=np.float64)
        positive = values >= 0.0
        out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        exp_values = np.exp(values[~positive])
        out[~positive] = exp_values / (1.0 + exp_values)
        return out
