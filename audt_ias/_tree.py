"""Sklearn-compatible unary classification density tree."""
import math
from dataclasses import dataclass
from typing import Any
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import KernelDensity
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
from ._loss import BCELogitLoss, density_from_probability, logit, make_loss
from ._splitter import (
    LOSS_BCE_APPROX,
    LOSS_BCE_CLOSED_FORM,
    LOSS_BCE_FIXED_POINT,
    apply_tree,
    find_best_split_bce,
    find_best_split_mse,
    find_random_split_bce,
    find_random_split_mse,
    predict_values,
)


@dataclass
class UnaryTree:
    """Parallel-array representation of a fitted unary density tree.

    Args:
        feature: Split feature for every node, or ``-1`` for leaves.
        threshold: Split threshold in normalized coordinates.
        value: Unary classifier value stored in every node.
        left: Left child index, or ``-1`` for leaves.
        right: Right child index, or ``-1`` for leaves.
        lower: Lower normalized box bounds for every node.
        upper: Upper normalized box bounds for every node.
        volume: Normalized unit-cube volume for every node box.
        n_node_samples: Number of training samples reaching every node.
    """

    feature: np.ndarray
    threshold: np.ndarray
    value: np.ndarray
    left: np.ndarray
    right: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    volume: np.ndarray
    n_node_samples: np.ndarray

    @property
    def node_count(self) -> int:
        """Return the number of nodes in the tree."""
        return self.value.shape[0]

    @property
    def leaf_mask(self) -> np.ndarray:
        """Return a boolean mask selecting leaf nodes."""
        return self.left == -1


class UnaryDensityTree(BaseEstimator):
    """Unary classification tree for nonparametric density estimation.

    The model learns a piecewise-constant classifier that separates observed
    samples from the uniform distribution in expectation. Densities are then
    reconstructed as ``phi(x) / (1 - phi(x))`` for the normalized unit cube and
    rescaled back to the original feature space.

    Args:
        max_depth: Maximum depth of the tree.
        min_samples_leaf: Minimum number of training samples in each leaf.
        splitter: Splitter strategy. Use ``"best"`` or ``"random"``. The
            random splitter samples one threshold uniformly from the in-node
            data range of each feature, then selects the feature with the best
            objective improvement.
        l2_reg: L2 regularization coefficient from the unary objective.
        lr: Learning rate multiplying the optimal split deltas.
        loss: Loss name. Use ``"mse"`` or ``"bce"``.
        bce_solver: BCE optimization variant. Use ``"approx"`` for the
            general second-order method, ``"closed_form"`` for the
            unregularized exact solution, or ``"fixed_point"`` for the
            regularized fixed-point variant.
        bce_fixed_point_iter: Number of fixed-point iterations for BCE.
        bounds: Optional original-space bounds with shape ``(2, n_features)``.
            If omitted, bounds are inferred from the training data.
        bounds_margin: Margin added to inferred bounds as a fraction of the
            feature range. Ignored when explicit ``bounds`` are provided.
        clip: Whether to clip transformed points to the learned bounds.
        leaf_value_mode: How leaf values are finalized. Use ``"path"`` to keep
            values accumulated along split paths, or ``"refit"`` to recompute
            leaf values from empirical leaf mass and leaf volume after the tree
            structure is built.
        leaf_mass_smoothing: Additive smoothing mass used when
            ``leaf_value_mode="refit"``. A value of ``0.0`` keeps the empirical
            mass-over-volume estimate; positive values add the same pseudo-count
            to every leaf.
        leaf_density: Leaf density scoring mode. Use ``"constant"`` for the
            unary tree's piecewise-constant density, or ``"kde"`` to fit local
            KDE models inside leaves after the tree structure is built.
        leaf_kde_bandwidth: Kernel bandwidth for ``leaf_density="kde"`` in
            leaf-local coordinates.
        leaf_kde_min_samples: Minimum samples required to fit a KDE in a leaf.
            Smaller leaves fall back to constant mass-over-volume density.
        leaf_kde_kernel: Kernel name passed to ``KernelDensity``.
        random_state: Random state used when ``splitter="random"``.
        eps: Numerical clipping value used for density reconstruction.
    """

    def __init__(
        self,
        max_depth: int = 14,
        min_samples_leaf: int = 5,
        splitter: str = "best",
        l2_reg: float = 0.0,
        lr: float = 1.0,
        loss: str = "bce",
        bce_solver: str = "closed_form",
        bce_fixed_point_iter: int = 5,
        bounds: np.ndarray | None = None,
        bounds_margin: float = 0.05,
        clip: bool = True,
        leaf_value_mode: str = "refit",
        leaf_mass_smoothing: float = 0.0,
        leaf_density: str = "constant",
        leaf_kde_bandwidth: float = 0.15,
        leaf_kde_min_samples: int = 10,
        leaf_kde_kernel: str = "gaussian",
        random_state: int | np.random.RandomState | None = None,
        eps: float = 1e-12,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.splitter = splitter
        self.l2_reg = l2_reg
        self.lr = lr
        self.loss = loss
        self.bce_solver = bce_solver
        self.bce_fixed_point_iter = bce_fixed_point_iter
        self.bounds = bounds
        self.bounds_margin = bounds_margin
        self.clip = clip
        self.leaf_value_mode = leaf_value_mode
        self.leaf_mass_smoothing = leaf_mass_smoothing
        self.leaf_density = leaf_density
        self.leaf_kde_bandwidth = leaf_kde_bandwidth
        self.leaf_kde_min_samples = leaf_kde_min_samples
        self.leaf_kde_kernel = leaf_kde_kernel
        self.random_state = random_state
        self.eps = eps

    def fit(self, X: np.ndarray, y: Any | None = None) -> "UnaryDensityTree":
        """Fit the unary density tree.

        Args:
            X: Training samples with shape ``(n_samples, n_features)``.
            y: Ignored target values, kept for scikit-learn compatibility.

        Returns:
            Fitted estimator.
        """
        del y
        self._validate_params()

        X_checked = check_array(X, dtype=np.float64, ensure_2d=True)
        X_normalized = self._fit_bounds_and_normalize(X_checked)
        X_normalized = np.ascontiguousarray(X_normalized, dtype=np.float64)

        self.loss_ = make_loss(
            self.loss,
            bce_solver=self.bce_solver,
            bce_fixed_point_iter=self.bce_fixed_point_iter,
            eps=self.eps,
        )
        self.init_value_ = self.loss_.initial_value
        self.n_features_in_ = X_normalized.shape[1]
        self.n_samples_fit_ = X_normalized.shape[0]

        depth_limited_nodes = 2 ** (self.max_depth + 1) - 1
        sample_limited_leaves = max(1, self.n_samples_fit_ // self.min_samples_leaf)
        sample_limited_nodes = 2 * sample_limited_leaves - 1
        max_possible_nodes = min(depth_limited_nodes, sample_limited_nodes)
        initial_capacity = min(max_possible_nodes, 1023)

        feature = np.full(initial_capacity, -1, dtype=np.int64)
        threshold = np.full(initial_capacity, np.nan, dtype=np.float64)
        value = np.full(initial_capacity, self.init_value_, dtype=np.float64)
        left = np.full(initial_capacity, -1, dtype=np.int64)
        right = np.full(initial_capacity, -1, dtype=np.int64)
        lower = np.zeros((initial_capacity, self.n_features_in_), dtype=np.float64)
        upper = np.ones((initial_capacity, self.n_features_in_), dtype=np.float64)
        volume = np.zeros(initial_capacity, dtype=np.float64)
        n_node_samples = np.zeros(initial_capacity, dtype=np.int64)

        def ensure_capacity(min_capacity: int) -> None:
            nonlocal feature
            nonlocal threshold
            nonlocal value
            nonlocal left
            nonlocal right
            nonlocal lower
            nonlocal upper
            nonlocal volume
            nonlocal n_node_samples

            old_capacity = feature.shape[0]
            if min_capacity <= old_capacity:
                return
            if min_capacity > max_possible_nodes:
                raise RuntimeError("Tree exceeded the maximum possible node count.")

            new_capacity = min(max_possible_nodes, max(min_capacity, 2 * old_capacity + 1))
            new_feature = np.full(new_capacity, -1, dtype=np.int64)
            new_threshold = np.full(new_capacity, np.nan, dtype=np.float64)
            new_value = np.full(new_capacity, self.init_value_, dtype=np.float64)
            new_left = np.full(new_capacity, -1, dtype=np.int64)
            new_right = np.full(new_capacity, -1, dtype=np.int64)
            new_lower = np.zeros((new_capacity, self.n_features_in_), dtype=np.float64)
            new_upper = np.ones((new_capacity, self.n_features_in_), dtype=np.float64)
            new_volume = np.zeros(new_capacity, dtype=np.float64)
            new_n_node_samples = np.zeros(new_capacity, dtype=np.int64)

            new_feature[:old_capacity] = feature
            new_threshold[:old_capacity] = threshold
            new_value[:old_capacity] = value
            new_left[:old_capacity] = left
            new_right[:old_capacity] = right
            new_lower[:old_capacity] = lower
            new_upper[:old_capacity] = upper
            new_volume[:old_capacity] = volume
            new_n_node_samples[:old_capacity] = n_node_samples

            feature = new_feature
            threshold = new_threshold
            value = new_value
            left = new_left
            right = new_right
            lower = new_lower
            upper = new_upper
            volume = new_volume
            n_node_samples = new_n_node_samples

        volume[0] = 1.0
        indices = np.arange(self.n_samples_fit_, dtype=np.int64)
        stack: list[tuple[int, int, int, int]] = [(0, 0, self.n_samples_fit_, 0)]
        node_count = 1
        rng = check_random_state(self.random_state)

        while stack:
            node_id, start, end, depth = stack.pop()
            n_samples = end - start
            n_node_samples[node_id] = n_samples

            if depth >= self.max_depth or n_samples < 2 * self.min_samples_leaf:
                continue

            split = self._find_split(
                X_normalized=X_normalized,
                indices=indices,
                start=start,
                end=end,
                lower=lower[node_id],
                upper=upper[node_id],
                volume=volume[node_id],
                value=value[node_id],
                n_total_samples=self.n_samples_fit_,
                rng=rng,
            )
            split_feature, split_threshold, _, left_delta, right_delta = split
            if split_feature < 0:
                continue

            split_pos = self._partition_indices(
                X_normalized,
                indices,
                start,
                end,
                split_feature,
                split_threshold,
            )
            n_left = split_pos - start
            n_right = end - split_pos
            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue

            ensure_capacity(node_count + 2)
            left_id = node_count
            right_id = node_count + 1
            node_count += 2

            feature[node_id] = split_feature
            threshold[node_id] = split_threshold
            left[node_id] = left_id
            right[node_id] = right_id

            lower[left_id] = lower[node_id]
            upper[left_id] = upper[node_id]
            upper[left_id, split_feature] = split_threshold
            lower[right_id] = lower[node_id]
            upper[right_id] = upper[node_id]
            lower[right_id, split_feature] = split_threshold

            width = upper[node_id, split_feature] - lower[node_id, split_feature]
            left_volume = (
                (split_threshold - lower[node_id, split_feature]) / width
            ) * volume[node_id]
            volume[left_id] = left_volume
            volume[right_id] = volume[node_id] - left_volume

            value[left_id] = value[node_id] + self.lr * left_delta
            value[right_id] = value[node_id] + self.lr * right_delta
            n_node_samples[left_id] = n_left
            n_node_samples[right_id] = n_right

            stack.append((right_id, split_pos, end, depth + 1))
            stack.append((left_id, start, split_pos, depth + 1))

        if self.leaf_value_mode == "refit":
            self._refit_leaf_values(
                value=value,
                left=left,
                volume=volume,
                n_node_samples=n_node_samples,
                node_count=node_count,
            )

        self.tree_ = UnaryTree(
            feature=feature[:node_count].copy(),
            threshold=threshold[:node_count].copy(),
            value=value[:node_count].copy(),
            left=left[:node_count].copy(),
            right=right[:node_count].copy(),
            lower=lower[:node_count].copy(),
            upper=upper[:node_count].copy(),
            volume=volume[:node_count].copy(),
            n_node_samples=n_node_samples[:node_count].copy(),
        )
        if self.leaf_density == "kde":
            self._fit_leaf_kde_models(X_normalized)
        return self

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Predict unary classifier values ``phi(x)``.

        Args:
            X: Samples with shape ``(n_samples, n_features)``.

        Returns:
            Piecewise-constant classifier values.
        """
        check_is_fitted(self, "tree_")
        X_normalized = self._normalize_for_prediction(X)
        raw = np.empty(X_normalized.shape[0], dtype=np.float64)
        tree = self.tree_
        predict_values(
            X_normalized,
            tree.feature,
            tree.threshold,
            tree.value,
            tree.left,
            tree.right,
            raw,
        )
        return self.loss_.to_probability(raw)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return stored raw tree values before the loss link function.

        For MSE this equals ``predict_raw``. For BCE these values are logits.

        Args:
            X: Samples with shape ``(n_samples, n_features)``.

        Returns:
            Raw leaf values.
        """
        check_is_fitted(self, "tree_")
        X_normalized = self._normalize_for_prediction(X)
        out = np.empty(X_normalized.shape[0], dtype=np.float64)
        tree = self.tree_
        predict_values(
            X_normalized,
            tree.feature,
            tree.threshold,
            tree.value,
            tree.left,
            tree.right,
            out,
        )
        return out

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Estimate density values for samples.

        Args:
            X: Samples with shape ``(n_samples, n_features)``.

        Returns:
            Density estimates in the original feature space.
        """
        if self.leaf_density == "kde":
            return np.exp(self.score_log_samples(X))
        raw = self.predict_raw(X)
        unit_density = density_from_probability(raw, self.eps)
        return unit_density / self.bounds_volume_

    def score_log_samples(self, X: np.ndarray) -> np.ndarray:
        """Estimate log-density values for samples.

        Args:
            X: Samples with shape ``(n_samples, n_features)``.

        Returns:
            Log-density estimates in the original feature space.
        """
        if self.leaf_density == "kde":
            return self._score_leaf_kde_log_samples(X)
        raw = self.predict_raw(X)
        unit_density = density_from_probability(raw, self.eps)
        unit_density = np.maximum(unit_density, self.eps)
        return np.log(unit_density) - math.log(self.bounds_volume_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Estimate density values for samples.

        Args:
            X: Samples with shape ``(n_samples, n_features)``.

        Returns:
            Density estimates in the original feature space.
        """
        return self.score_samples(X)

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Return the leaf index for each sample.

        Args:
            X: Samples with shape ``(n_samples, n_features)``.

        Returns:
            Leaf node indices.
        """
        check_is_fitted(self, "tree_")
        X_normalized = self._normalize_for_prediction(X)
        out = np.empty(X_normalized.shape[0], dtype=np.int64)
        tree = self.tree_
        apply_tree(
            X_normalized,
            tree.feature,
            tree.threshold,
            tree.left,
            tree.right,
            out,
        )
        return out

    def leaf_boxes(self, original_scale: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return leaf boxes and leaf values.

        Args:
            original_scale: Whether to convert boxes to the original feature scale.

        Returns:
            Tuple ``(lower, upper, value)`` for all leaves.
        """
        check_is_fitted(self, "tree_")
        mask = self.tree_.leaf_mask
        lower = self.tree_.lower[mask].copy()
        upper = self.tree_.upper[mask].copy()
        values = self.tree_.value[mask].copy()
        values = self.loss_.to_probability(values)
        if original_scale:
            lower = lower * self.bounds_scale_ + self.bounds_min_
            upper = upper * self.bounds_scale_ + self.bounds_min_
        return lower, upper, values

    def _validate_params(self) -> None:
        if not isinstance(self.max_depth, int) or self.max_depth < 0:
            raise ValueError("max_depth must be a non-negative integer.")
        if self.max_depth > 30:
            raise ValueError("max_depth > 30 would allocate too many tree nodes.")
        if not isinstance(self.min_samples_leaf, int) or self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be a positive integer.")
        if self.splitter not in {"best", "random"}:
            raise ValueError("splitter must be either 'best' or 'random'.")
        if self.loss not in {"mse", "bce"}:
            raise ValueError("loss must be either 'mse' or 'bce'.")
        if self.bce_solver not in {"approx", "closed_form", "fixed_point"}:
            raise ValueError("bce_solver must be 'approx', 'closed_form', or 'fixed_point'.")
        if not isinstance(self.bce_fixed_point_iter, int) or self.bce_fixed_point_iter < 1:
            raise ValueError("bce_fixed_point_iter must be a positive integer.")
        if self.loss == "bce" and self.bce_solver == "closed_form" and self.l2_reg != 0.0:
            raise ValueError("bce_solver='closed_form' requires l2_reg=0.")
        if self.l2_reg < 0.0:
            raise ValueError("l2_reg must be non-negative.")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive.")
        if self.bounds_margin < 0.0:
            raise ValueError("bounds_margin must be non-negative.")
        if self.leaf_value_mode not in {"path", "refit"}:
            raise ValueError("leaf_value_mode must be 'path' or 'refit'.")
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
        if self.eps <= 0.0 or self.eps >= 0.5:
            raise ValueError("eps must be in (0, 0.5).")

    def _refit_leaf_values(
        self,
        value: np.ndarray,
        left: np.ndarray,
        volume: np.ndarray,
        n_node_samples: np.ndarray,
        node_count: int,
    ) -> None:
        leaf_mask = left[:node_count] == -1
        leaf_ids = np.flatnonzero(leaf_mask)
        alpha = float(self.leaf_mass_smoothing)
        denominator = self.n_samples_fit_ + alpha * leaf_ids.shape[0]
        fractions = (n_node_samples[leaf_ids] + alpha) / denominator
        probabilities = fractions / (fractions + volume[leaf_ids])
        probabilities = np.clip(probabilities, self.eps, 1.0 - self.eps)
        if isinstance(self.loss_, BCELogitLoss):
            value[leaf_ids] = np.array(
                [logit(float(probability), self.eps) for probability in probabilities],
                dtype=np.float64,
            )
        else:
            value[leaf_ids] = probabilities

    def _fit_leaf_kde_models(self, X_normalized: np.ndarray) -> None:
        tree = self.tree_
        leaves = np.empty(X_normalized.shape[0], dtype=np.int64)
        apply_tree(
            X_normalized,
            tree.feature,
            tree.threshold,
            tree.left,
            tree.right,
            leaves,
        )

        leaf_ids = np.flatnonzero(tree.leaf_mask)
        alpha = float(self.leaf_mass_smoothing)
        denominator = self.n_samples_fit_ + alpha * leaf_ids.shape[0]
        self.leaf_log_mass_ = np.full(tree.node_count, -np.inf, dtype=np.float64)
        self.leaf_log_volume_ = np.full(tree.node_count, -np.inf, dtype=np.float64)
        self.leaf_kde_models_: dict[int, KernelDensity] = {}

        for leaf_id in leaf_ids:
            count = int(tree.n_node_samples[leaf_id])
            mass = (count + alpha) / denominator
            volume = max(float(tree.volume[leaf_id]), self.eps)
            self.leaf_log_mass_[leaf_id] = math.log(max(mass, self.eps))
            self.leaf_log_volume_[leaf_id] = math.log(volume)
            if count < self.leaf_kde_min_samples:
                continue
            mask = leaves == leaf_id
            local = self._to_leaf_local(X_normalized[mask], leaf_id)
            model = KernelDensity(
                bandwidth=self.leaf_kde_bandwidth,
                kernel=self.leaf_kde_kernel,
            )
            self.leaf_kde_models_[int(leaf_id)] = model.fit(local)

    def _score_leaf_kde_log_samples(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "leaf_log_mass_")
        X_normalized = self._normalize_for_prediction(X)
        leaves = np.empty(X_normalized.shape[0], dtype=np.int64)
        tree = self.tree_
        apply_tree(
            X_normalized,
            tree.feature,
            tree.threshold,
            tree.left,
            tree.right,
            leaves,
        )
        out = np.empty(X_normalized.shape[0], dtype=np.float64)
        for leaf_id in np.unique(leaves):
            mask = leaves == leaf_id
            base = self.leaf_log_mass_[leaf_id] - self.leaf_log_volume_[leaf_id]
            model = self.leaf_kde_models_.get(int(leaf_id))
            if model is None:
                out[mask] = base
                continue
            local = self._to_leaf_local(X_normalized[mask], int(leaf_id))
            out[mask] = (
                self.leaf_log_mass_[leaf_id]
                + model.score_samples(local)
                - self.leaf_log_volume_[leaf_id]
            )
        return out - math.log(self.bounds_volume_)

    def _to_leaf_local(self, X_normalized: np.ndarray, leaf_id: int) -> np.ndarray:
        lower = self.tree_.lower[leaf_id]
        upper = self.tree_.upper[leaf_id]
        width = np.maximum(upper - lower, self.eps)
        local = (X_normalized - lower) / width
        return np.ascontiguousarray(np.clip(local, 0.0, 1.0), dtype=np.float64)

    def _fit_bounds_and_normalize(self, X: np.ndarray) -> np.ndarray:
        if self.bounds is None:
            data_min = np.min(X, axis=0)
            data_max = np.max(X, axis=0)
            data_range = data_max - data_min
            margin = self.bounds_margin * data_range
            bounds_min = data_min - margin
            bounds_max = data_max + margin
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

        self.bounds_min_ = bounds_min
        self.bounds_max_ = bounds_max
        self.bounds_ = np.vstack([bounds_min, bounds_max])
        self.bounds_scale_ = scale
        self.bounds_volume_ = float(np.prod(scale))
        return self._normalize_array(X)

    def _normalize_for_prediction(self, X: np.ndarray) -> np.ndarray:
        X_checked = check_array(X, dtype=np.float64, ensure_2d=True)
        if X_checked.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_checked.shape[1]} features, expected {self.n_features_in_}."
            )
        return np.ascontiguousarray(self._normalize_array(X_checked), dtype=np.float64)

    def _normalize_array(self, X: np.ndarray) -> np.ndarray:
        X_normalized = (X - self.bounds_min_) / self.bounds_scale_
        if self.clip:
            X_normalized = np.clip(X_normalized, 0.0, 1.0)
        elif np.any((X_normalized < 0.0) | (X_normalized > 1.0)):
            raise ValueError("X contains points outside the fitted bounds.")
        return X_normalized

    def _find_split(
        self,
        X_normalized: np.ndarray,
        indices: np.ndarray,
        start: int,
        end: int,
        lower: np.ndarray,
        upper: np.ndarray,
        volume: float,
        value: float,
        n_total_samples: int,
        rng: np.random.RandomState,
    ) -> tuple[int, float, float, float, float]:
        if self.splitter == "best":
            if isinstance(self.loss_, BCELogitLoss):
                return find_best_split_bce(
                    X_normalized,
                    indices,
                    start,
                    end,
                    lower,
                    upper,
                    volume,
                    value,
                    self.min_samples_leaf,
                    self.l2_reg,
                    self._bce_loss_code(),
                    self.bce_fixed_point_iter,
                    self.eps,
                    n_total_samples,
                )
            else:
                return find_best_split_mse(
                    X_normalized,
                    indices,
                    start,
                    end,
                    lower,
                    upper,
                    volume,
                    value,
                    self.min_samples_leaf,
                    self.l2_reg,
                    n_total_samples,
                )
        return self._find_random_split(
            X_normalized,
            indices,
            start,
            end,
            lower,
            upper,
            volume,
            value,
            n_total_samples,
            rng,
        )

    def _find_random_split(
        self,
        X_normalized: np.ndarray,
        indices: np.ndarray,
        start: int,
        end: int,
        lower: np.ndarray,
        upper: np.ndarray,
        volume: float,
        value: float,
        n_total_samples: int,
        rng: np.random.RandomState,
    ) -> tuple[int, float, float, float, float]:
        random_unit = rng.random_sample(X_normalized.shape[1]).astype(np.float64)
        if isinstance(self.loss_, BCELogitLoss):
            return find_random_split_bce(
                X_normalized,
                indices,
                start,
                end,
                lower,
                upper,
                volume,
                value,
                self.min_samples_leaf,
                self.l2_reg,
                self._bce_loss_code(),
                self.bce_fixed_point_iter,
                self.eps,
                n_total_samples,
                random_unit,
            )
        return find_random_split_mse(
            X_normalized,
            indices,
            start,
            end,
            lower,
            upper,
            volume,
            value,
            self.min_samples_leaf,
            self.l2_reg,
            n_total_samples,
            random_unit,
        )

    def _bce_loss_code(self) -> int:
        if self.bce_solver == "approx":
            return LOSS_BCE_APPROX
        if self.bce_solver == "closed_form":
            return LOSS_BCE_CLOSED_FORM
        return LOSS_BCE_FIXED_POINT

    @staticmethod
    def _partition_indices(
        X: np.ndarray,
        indices: np.ndarray,
        start: int,
        end: int,
        feature: int,
        threshold: float,
    ) -> int:
        left_pos = start
        right_pos = end - 1
        while left_pos <= right_pos:
            sample_idx = indices[left_pos]
            if X[sample_idx, feature] <= threshold:
                left_pos += 1
            else:
                indices[left_pos], indices[right_pos] = indices[right_pos], sample_idx
                right_pos -= 1
        return left_pos
