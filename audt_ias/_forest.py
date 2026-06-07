"""Ensembles of unary density trees."""
from dataclasses import dataclass
from typing import Any
import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
from ._tree import UnaryDensityTree


@dataclass
class _TreeTransform:
    """Per-estimator feature subset and linear transform."""

    features: np.ndarray
    center: np.ndarray
    rotation: np.ndarray


@dataclass
class _TreeSpec:
    """Sampling and transform specification for one tree."""

    sample_indices: np.ndarray
    features: np.ndarray
    transform: _TreeTransform
    tree_bounds: np.ndarray
    random_state: int


def _fit_single_tree(
    X: np.ndarray,
    spec: _TreeSpec,
    tree_params: dict[str, Any],
) -> UnaryDensityTree:
    X_sampled = X[spec.sample_indices][:, spec.features]
    X_tree = UnaryDensityForest._apply_transform(X_sampled, spec.transform)
    tree = UnaryDensityTree(
        bounds=spec.tree_bounds,
        bounds_margin=0.0,
        random_state=spec.random_state,
        **tree_params,
    )
    return tree.fit(X_tree)


class UnaryDensityForest(BaseEstimator):
    """Bagged ensemble of unary density trees.

    Each tree is fitted on a bootstrap or subsampled set of rows and on a
    per-tree random subspace of columns. Optionally, each selected feature
    subspace is transformed by a tree-specific orthogonal rotation before the
    unary density tree is fitted.

    Args:
        n_estimators: Number of trees in the ensemble.
        max_samples: Number or fraction of samples used for each tree. ``None``
            means all samples.
        bootstrap: Whether to sample rows with replacement.
        max_features: Number, fraction, or rule for features used by each tree.
            Supports ``None``, ``"sqrt"``, and ``"log2"``.
        bootstrap_features: Whether to sample features with replacement.
        rotation: Per-tree feature transform. Use ``False`` or ``"none"`` for
            no transform, ``True`` or ``"random"`` for random orthogonal
            rotations, and ``"pca"`` for a PCA-style rotation fitted on each
            tree's sampled data.
        rotation_bounds: Bounds used after feature rotation. ``"domain_aabb"``
            uses the axis-aligned bounding box of the transformed modeling
            domain and preserves the previous behavior. ``"data_aabb"`` uses
            transformed full-training data bounds for each tree, and
            ``"bootstrap_aabb"`` uses transformed sampled rows.
        max_depth: Maximum depth passed to each tree.
        min_samples_leaf: Minimum leaf size passed to each tree.
        splitter: Splitter strategy passed to each tree.
        l2_reg: L2 regularization coefficient passed to each tree.
        lr: Learning rate passed to each tree.
        loss: Loss name passed to each tree.
        bce_solver: BCE optimization variant passed to each tree.
        bce_fixed_point_iter: Number of BCE fixed-point iterations.
        leaf_value_mode: Leaf value finalization strategy passed to each tree.
        leaf_mass_smoothing: Additive smoothing mass passed to tree refit.
        leaf_density: Leaf density scoring mode passed to each tree.
        leaf_kde_bandwidth: Leaf-local KDE bandwidth passed to each tree.
        leaf_kde_min_samples: Minimum leaf samples for local KDE fitting.
        leaf_kde_kernel: Kernel name used for local leaf KDE.
        bounds: Optional original-space bounds with shape ``(2, n_features)``.
            If omitted, bounds are inferred from the full training data before
            row subsampling. Per-tree transformed bounds are derived from this
            domain, so rotated trees do not clip valid domain points.
        bounds_margin: Margin added to inferred original-space bounds as a
            fraction of the feature range. Ignored when explicit ``bounds`` are
            provided.
        clip: Whether each tree clips transformed points to its fitted bounds.
        density_aggregation: How per-tree densities are aggregated. Use
            ``"mean_density"`` for the current arithmetic mean of densities, or
            ``"mean_log_density"`` for the geometric mean of per-tree densities.
        n_jobs: Number of parallel jobs used for tree fitting. ``None`` and
            ``1`` use sequential fitting. ``-1`` uses all available workers.
        random_state: Random seed.
        eps: Numerical clipping value passed to each tree.
    """

    def __init__(
        self,
        n_estimators: int = 256,
        max_samples: int | float | None = 0.8,
        bootstrap: bool = True,
        max_features: int | float | str | None = 1.0,
        bootstrap_features: bool = False,
        rotation: bool | str = "random",
        rotation_bounds: str = "domain_aabb",
        max_depth: int = 14,
        min_samples_leaf: int = 3,
        splitter: str = "best",
        l2_reg: float = 1e-3,
        lr: float = 1.0,
        loss: str = "mse",
        bce_solver: str = "approx",
        bce_fixed_point_iter: int = 5,
        leaf_value_mode: str = "refit",
        leaf_mass_smoothing: float = 0.0,
        leaf_density: str = "constant",
        leaf_kde_bandwidth: float = 0.15,
        leaf_kde_min_samples: int = 10,
        leaf_kde_kernel: str = "gaussian",
        bounds: np.ndarray | None = None,
        bounds_margin: float = 0.05,
        clip: bool = True,
        density_aggregation: str = "mean_density",
        n_jobs: int | None = None,
        random_state: int | np.random.RandomState | None = None,
        eps: float = 1e-12,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.max_features = max_features
        self.bootstrap_features = bootstrap_features
        self.rotation = rotation
        self.rotation_bounds = rotation_bounds
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.splitter = splitter
        self.l2_reg = l2_reg
        self.lr = lr
        self.loss = loss
        self.bce_solver = bce_solver
        self.bce_fixed_point_iter = bce_fixed_point_iter
        self.leaf_value_mode = leaf_value_mode
        self.leaf_mass_smoothing = leaf_mass_smoothing
        self.leaf_density = leaf_density
        self.leaf_kde_bandwidth = leaf_kde_bandwidth
        self.leaf_kde_min_samples = leaf_kde_min_samples
        self.leaf_kde_kernel = leaf_kde_kernel
        self.bounds = bounds
        self.bounds_margin = bounds_margin
        self.clip = clip
        self.density_aggregation = density_aggregation
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.eps = eps

    def fit(self, X: np.ndarray, y: Any | None = None) -> "UnaryDensityForest":
        """Fit the forest.

        Args:
            X: Training samples with shape ``(n_samples, n_features)``.
            y: Ignored target values, kept for scikit-learn compatibility.

        Returns:
            Fitted estimator.
        """
        del y
        self._validate_params()
        X_checked = check_array(X, dtype=np.float64, ensure_2d=True)
        rng = check_random_state(self.random_state)

        self.n_features_in_ = X_checked.shape[1]
        self.n_samples_fit_ = X_checked.shape[0]
        self.max_samples_ = self._resolve_max_samples(self.n_samples_fit_)
        self.max_features_ = self._resolve_max_features(self.n_features_in_)
        self.rotation_ = self._resolve_rotation()
        self.bounds_ = self._fit_bounds(X_checked)

        specs: list[_TreeSpec] = []
        for _ in range(self.n_estimators):
            sample_indices = self._sample_rows(rng, self.n_samples_fit_)
            features = self._sample_features(rng, self.n_features_in_)
            X_subset = X_checked[:, features]
            X_sampled = X_subset[sample_indices]
            transform = self._make_transform(X_sampled, features, rng)
            tree_bounds = self._make_tree_bounds(
                X_subset=X_subset,
                X_sampled=X_sampled,
                original_bounds=self.bounds_[:, features],
                transform=transform,
            )
            specs.append(
                _TreeSpec(
                    sample_indices=sample_indices,
                    features=features,
                    transform=transform,
                    tree_bounds=tree_bounds,
                    random_state=int(rng.randint(np.iinfo(np.int32).max)),
                )
            )

        tree_params = {
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "splitter": self.splitter,
            "l2_reg": self.l2_reg,
            "lr": self.lr,
            "loss": self.loss,
            "bce_solver": self.bce_solver,
            "bce_fixed_point_iter": self.bce_fixed_point_iter,
            "leaf_value_mode": self.leaf_value_mode,
            "leaf_mass_smoothing": self.leaf_mass_smoothing,
            "leaf_density": self.leaf_density,
            "leaf_kde_bandwidth": self.leaf_kde_bandwidth,
            "leaf_kde_min_samples": self.leaf_kde_min_samples,
            "leaf_kde_kernel": self.leaf_kde_kernel,
            "clip": self.clip,
            "eps": self.eps,
        }
        self.n_jobs_ = effective_n_jobs(self.n_jobs)
        if self.n_jobs_ == 1:
            estimators = [
                _fit_single_tree(X_checked, spec, tree_params) for spec in specs
            ]
        else:
            estimators = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_fit_single_tree)(X_checked, spec, tree_params)
                for spec in specs
            )

        self.estimators_ = estimators
        self.estimators_samples_ = [spec.sample_indices for spec in specs]
        self.estimators_features_ = [spec.features for spec in specs]
        self.transforms_ = [spec.transform for spec in specs]

        return self

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Predict averaged unary classifier values.

        Args:
            X: Samples with shape ``(n_samples, n_features)``.

        Returns:
            Mean classifier value across trees.
        """
        check_is_fitted(self, "estimators_")
        X_checked = self._validate_X_predict(X)
        out = np.zeros(X_checked.shape[0], dtype=np.float64)
        for tree, transform in zip(self.estimators_, self.transforms_):
            out += tree.predict_raw(self._transform_X(X_checked, transform))
        out /= len(self.estimators_)
        return out

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Estimate density values by averaging per-tree densities.

        Args:
            X: Samples with shape ``(n_samples, n_features)``.

        Returns:
            Mean density estimate across trees. If ``max_features`` selects a
            proper subspace, these are averaged subspace density estimates.
        """
        check_is_fitted(self, "estimators_")
        X_checked = self._validate_X_predict(X)
        if self.density_aggregation == "mean_log_density":
            return np.exp(self.score_log_samples(X_checked))

        out = np.zeros(X_checked.shape[0], dtype=np.float64)
        for tree, transform in zip(self.estimators_, self.transforms_):
            out += tree.score_samples(self._transform_X(X_checked, transform))
        out /= len(self.estimators_)
        return out

    def score_log_samples(self, X: np.ndarray) -> np.ndarray:
        """Estimate log-density values by aggregating per-tree log densities.

        Args:
            X: Samples with shape ``(n_samples, n_features)``.

        Returns:
            Log-density estimate across trees.
        """
        check_is_fitted(self, "estimators_")
        X_checked = self._validate_X_predict(X)
        if self.density_aggregation == "mean_log_density":
            out = np.zeros(X_checked.shape[0], dtype=np.float64)
            for tree, transform in zip(self.estimators_, self.transforms_):
                out += tree.score_log_samples(self._transform_X(X_checked, transform))
            out /= len(self.estimators_)
            return out

        out = None
        for tree, transform in zip(self.estimators_, self.transforms_):
            log_density = tree.score_log_samples(self._transform_X(X_checked, transform))
            if out is None:
                out = log_density
            else:
                out = np.logaddexp(out, log_density)
        if out is None:
            raise RuntimeError("No fitted estimators.")
        return out - np.log(len(self.estimators_))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Estimate density values for samples.

        Args:
            X: Samples with shape ``(n_samples, n_features)``.

        Returns:
            Mean density estimate across trees.
        """
        return self.score_samples(X)

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Return per-tree leaf indices for samples.

        Args:
            X: Samples with shape ``(n_samples, n_features)``.

        Returns:
            Matrix with shape ``(n_samples, n_estimators)``.
        """
        check_is_fitted(self, "estimators_")
        X_checked = self._validate_X_predict(X)
        leaves = np.empty((X_checked.shape[0], len(self.estimators_)), dtype=np.int64)
        for j, (tree, transform) in enumerate(zip(self.estimators_, self.transforms_)):
            leaves[:, j] = tree.apply(self._transform_X(X_checked, transform))
        return leaves

    def _validate_params(self) -> None:
        if not isinstance(self.n_estimators, int) or self.n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer.")
        if not isinstance(self.bootstrap, bool):
            raise ValueError("bootstrap must be boolean.")
        if not isinstance(self.bootstrap_features, bool):
            raise ValueError("bootstrap_features must be boolean.")
        if self.bounds_margin < 0.0:
            raise ValueError("bounds_margin must be non-negative.")
        if self.rotation_bounds not in {"domain_aabb", "data_aabb", "bootstrap_aabb"}:
            raise ValueError(
                "rotation_bounds must be 'domain_aabb', 'data_aabb', or "
                "'bootstrap_aabb'."
            )
        if self.leaf_mass_smoothing < 0.0:
            raise ValueError("leaf_mass_smoothing must be non-negative.")
        if self.leaf_density not in {"constant", "kde"}:
            raise ValueError("leaf_density must be 'constant' or 'kde'.")
        if self.leaf_kde_bandwidth <= 0.0:
            raise ValueError("leaf_kde_bandwidth must be positive.")
        if (
            not isinstance(self.leaf_kde_min_samples, int)
            or self.leaf_kde_min_samples < 1
        ):
            raise ValueError("leaf_kde_min_samples must be a positive integer.")
        if self.density_aggregation not in {"mean_density", "mean_log_density"}:
            raise ValueError(
                "density_aggregation must be 'mean_density' or 'mean_log_density'."
            )
        if self.n_jobs is not None:
            if not isinstance(self.n_jobs, int):
                raise ValueError("n_jobs must be an integer or None.")
            if self.n_jobs == 0:
                raise ValueError("n_jobs must not be 0.")

    def _fit_bounds(self, X: np.ndarray) -> np.ndarray:
        if self.bounds is None:
            bounds_min = np.min(X, axis=0)
            bounds_max = np.max(X, axis=0)
            data_range = bounds_max - bounds_min
            margin = self.bounds_margin * data_range
            bounds_min = bounds_min - margin
            bounds_max = bounds_max + margin
        else:
            bounds = np.asarray(self.bounds, dtype=np.float64)
            if bounds.shape != (2, X.shape[1]):
                raise ValueError("bounds must have shape (2, n_features).")
            bounds_min = bounds[0].copy()
            bounds_max = bounds[1].copy()

        scale = bounds_max - bounds_min
        zero_width = scale <= 0.0
        if np.any(zero_width):
            scale[zero_width] = 1.0
            bounds_max = bounds_min + scale
        return np.vstack([bounds_min, bounds_max])

    @staticmethod
    def _transform_bounds(bounds: np.ndarray, transform: _TreeTransform) -> np.ndarray:
        lower = bounds[0] - transform.center
        upper = bounds[1] - transform.center
        rotation = transform.rotation

        out_lower = np.empty(rotation.shape[1], dtype=np.float64)
        out_upper = np.empty(rotation.shape[1], dtype=np.float64)
        for j in range(rotation.shape[1]):
            coeff = rotation[:, j]
            low_contrib = np.where(coeff >= 0.0, coeff * lower, coeff * upper)
            high_contrib = np.where(coeff >= 0.0, coeff * upper, coeff * lower)
            out_lower[j] = np.sum(low_contrib)
            out_upper[j] = np.sum(high_contrib)
        return np.vstack([out_lower, out_upper])

    def _make_tree_bounds(
        self,
        X_subset: np.ndarray,
        X_sampled: np.ndarray,
        original_bounds: np.ndarray,
        transform: _TreeTransform,
    ) -> np.ndarray:
        if self.rotation_bounds == "domain_aabb":
            return self._transform_bounds(original_bounds, transform)
        if self.rotation_bounds == "data_aabb":
            X_bounds = self._apply_transform(X_subset, transform)
        else:
            X_bounds = self._apply_transform(X_sampled, transform)
        return self._bounds_from_transformed_data(X_bounds)

    @staticmethod
    def _bounds_from_transformed_data(X: np.ndarray) -> np.ndarray:
        bounds_min = np.min(X, axis=0)
        bounds_max = np.max(X, axis=0)
        scale = bounds_max - bounds_min
        zero_width = scale <= 0.0
        if np.any(zero_width):
            scale[zero_width] = 1.0
            bounds_max = bounds_min + scale
        return np.vstack([bounds_min, bounds_max])

    def _resolve_max_samples(self, n_samples: int) -> int:
        if self.max_samples is None:
            return n_samples
        if isinstance(self.max_samples, int):
            if self.max_samples < 1:
                raise ValueError("max_samples must be positive.")
            return min(self.max_samples, n_samples)
        if isinstance(self.max_samples, float):
            if self.max_samples <= 0.0 or self.max_samples > 1.0:
                raise ValueError("float max_samples must be in (0, 1].")
            return max(1, int(round(self.max_samples * n_samples)))
        raise ValueError("max_samples must be an int, float, or None.")

    def _resolve_max_features(self, n_features: int) -> int:
        if self.max_features is None:
            return n_features
        if isinstance(self.max_features, int):
            if self.max_features < 1:
                raise ValueError("max_features must be positive.")
            return min(self.max_features, n_features)
        if isinstance(self.max_features, float):
            if self.max_features <= 0.0 or self.max_features > 1.0:
                raise ValueError("float max_features must be in (0, 1].")
            return max(1, int(round(self.max_features * n_features)))
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        raise ValueError("Unsupported max_features value.")

    def _resolve_rotation(self) -> str:
        if self.rotation is True:
            return "random"
        if self.rotation is False or self.rotation is None:
            return "none"
        if self.rotation in {"none", "random", "pca"}:
            return str(self.rotation)
        raise ValueError("rotation must be False, True, 'none', 'random', or 'pca'.")

    def _sample_rows(self, rng: np.random.RandomState, n_samples: int) -> np.ndarray:
        if self.bootstrap:
            return rng.randint(0, n_samples, size=self.max_samples_).astype(np.int64)
        return rng.permutation(n_samples)[: self.max_samples_].astype(np.int64)

    def _sample_features(self, rng: np.random.RandomState, n_features: int) -> np.ndarray:
        if self.bootstrap_features:
            return rng.randint(0, n_features, size=self.max_features_).astype(np.int64)
        return rng.choice(n_features, size=self.max_features_, replace=False).astype(np.int64)

    def _make_transform(
        self,
        X_sampled: np.ndarray,
        features: np.ndarray,
        rng: np.random.RandomState,
    ) -> _TreeTransform:
        n_features = X_sampled.shape[1]
        if self.rotation_ == "none" or n_features == 1:
            center = np.zeros(n_features, dtype=np.float64)
            rotation = np.eye(n_features, dtype=np.float64)
        elif self.rotation_ == "random":
            center = np.mean(X_sampled, axis=0)
            random_matrix = rng.normal(size=(n_features, n_features))
            rotation, _ = np.linalg.qr(random_matrix)
        else:
            center = np.mean(X_sampled, axis=0)
            _, _, vt = np.linalg.svd(X_sampled - center, full_matrices=False)
            rotation = vt.T
        return _TreeTransform(features=features, center=center, rotation=rotation)

    @staticmethod
    def _apply_transform(X: np.ndarray, transform: _TreeTransform) -> np.ndarray:
        centered = X - transform.center
        transformed = np.einsum("ij,jk->ik", centered, transform.rotation)
        return np.ascontiguousarray(transformed)

    def _transform_X(self, X: np.ndarray, transform: _TreeTransform) -> np.ndarray:
        return self._apply_transform(X[:, transform.features], transform)

    def _validate_X_predict(self, X: np.ndarray) -> np.ndarray:
        X_checked = check_array(X, dtype=np.float64, ensure_2d=True)
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_checked.shape[1]} features, expected {self.n_features_in_}."
            )
        return X_checked
