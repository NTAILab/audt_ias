"""Numerical split search kernels."""
import numpy as np


LOSS_MSE = 0
LOSS_BCE_APPROX = 1
LOSS_BCE_CLOSED_FORM = 2
LOSS_BCE_FIXED_POINT = 3


def _sigmoid_scalar(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + np.exp(-value))
    exp_value = np.exp(value)
    return exp_value / (1.0 + exp_value)


def _logit_scalar(probability: float, eps: float) -> float:
    clipped = min(max(probability, eps), 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def _softplus_scalar(value: float) -> float:
    if value > 0.0:
        return value + np.log1p(np.exp(-value))
    return np.log1p(np.exp(value))


def _bce_approx_delta(
    value: float,
    volume: float,
    fraction: float,
    l2_reg: float,
) -> float:
    probability = _sigmoid_scalar(value)
    hessian = probability * (1.0 - probability)
    linear = probability * volume + (probability - 1.0) * fraction
    quadratic = hessian * (volume + fraction) + l2_reg
    return -linear / quadratic


def _bce_fixed_point_delta(
    value: float,
    volume: float,
    fraction: float,
    l2_reg: float,
    bce_fixed_point_iter: int,
) -> float:
    delta = _bce_approx_delta(value, volume, fraction, l2_reg)
    for _ in range(bce_fixed_point_iter):
        probability = _sigmoid_scalar(value + delta)
        delta = (fraction - probability * (volume + fraction)) / l2_reg
    return delta


def _bce_delta(
    value: float,
    volume: float,
    fraction: float,
    l2_reg: float,
    loss_code: int,
    bce_fixed_point_iter: int,
    eps: float,
) -> float:
    if loss_code == LOSS_BCE_APPROX:
        return _bce_approx_delta(value, volume, fraction, l2_reg)
    if loss_code == LOSS_BCE_CLOSED_FORM or l2_reg == 0.0:
        probability = fraction / (volume + fraction)
        return _logit_scalar(probability, eps) - value
    return _bce_fixed_point_delta(
        value,
        volume,
        fraction,
        l2_reg,
        bce_fixed_point_iter,
    )


def _bce_score(
    value: float,
    left_volume: float,
    right_volume: float,
    left_fraction: float,
    right_fraction: float,
    left_delta: float,
    right_delta: float,
    l2_reg: float,
    loss_code: int,
) -> float:
    if loss_code == LOSS_BCE_APPROX:
        probability = _sigmoid_scalar(value)
        hessian = probability * (1.0 - probability)
        left_linear = probability * left_volume + (probability - 1.0) * left_fraction
        right_linear = probability * right_volume + (probability - 1.0) * right_fraction
        left_quadratic = hessian * (left_volume + left_fraction) + l2_reg
        right_quadratic = hessian * (right_volume + right_fraction) + l2_reg
        score = left_linear * left_linear / left_quadratic
        score += right_linear * right_linear / right_quadratic
        return score

    left_value = value + left_delta
    right_value = value + right_delta
    objective = left_volume * _softplus_scalar(left_value)
    objective += left_fraction * _softplus_scalar(-left_value)
    objective += 0.5 * l2_reg * left_delta * left_delta
    objective += right_volume * _softplus_scalar(right_value)
    objective += right_fraction * _softplus_scalar(-right_value)
    objective += 0.5 * l2_reg * right_delta * right_delta
    return -objective


def _find_best_split_mse_py(
    X: np.ndarray,
    indices: np.ndarray,
    start: int,
    end: int,
    lower: np.ndarray,
    upper: np.ndarray,
    volume: float,
    value: float,
    min_samples_leaf: int,
    l2_reg: float,
    n_total_samples: int,
) -> tuple[int, float, float, float, float]:
    n_samples = end - start
    n_features = X.shape[1]
    best_score = -1.0
    best_feature = -1
    best_threshold = 0.0
    best_left_delta = 0.0
    best_right_delta = 0.0

    values = np.empty(n_samples, dtype=np.float64)

    for feature in range(n_features):
        width = upper[feature] - lower[feature]
        if width <= 0.0:
            continue

        for i in range(n_samples):
            values[i] = X[indices[start + i], feature]

        order = np.argsort(values)
        for pos in range(min_samples_leaf, n_samples - min_samples_leaf + 1):
            left_value = values[order[pos - 1]]
            right_value = values[order[pos]]
            if left_value == right_value:
                continue

            threshold = 0.5 * (left_value + right_value)
            if threshold <= lower[feature] or threshold >= upper[feature]:
                continue

            n_left = pos
            n_right = n_samples - pos
            left_volume = ((threshold - lower[feature]) / width) * volume
            right_volume = volume - left_volume

            left_fraction = n_left / n_total_samples
            right_fraction = n_right / n_total_samples
            left_linear = value * left_volume + (value - 1.0) * left_fraction
            right_linear = value * right_volume + (value - 1.0) * right_fraction
            left_quad = left_volume + left_fraction + l2_reg
            right_quad = right_volume + right_fraction + l2_reg

            score = left_linear * left_linear / left_quad
            score += right_linear * right_linear / right_quad
            if score > best_score:
                best_score = score
                best_feature = feature
                best_threshold = threshold
                best_left_delta = -left_linear / left_quad
                best_right_delta = -right_linear / right_quad

    return (
        best_feature,
        best_threshold,
        best_score,
        best_left_delta,
        best_right_delta,
    )


def _find_best_split_bce_py(
    X: np.ndarray,
    indices: np.ndarray,
    start: int,
    end: int,
    lower: np.ndarray,
    upper: np.ndarray,
    volume: float,
    value: float,
    min_samples_leaf: int,
    l2_reg: float,
    loss_code: int,
    bce_fixed_point_iter: int,
    eps: float,
    n_total_samples: int,
) -> tuple[int, float, float, float, float]:
    n_samples = end - start
    n_features = X.shape[1]
    best_score = -np.inf
    best_feature = -1
    best_threshold = 0.0
    best_left_delta = 0.0
    best_right_delta = 0.0

    values = np.empty(n_samples, dtype=np.float64)

    for feature in range(n_features):
        width = upper[feature] - lower[feature]
        if width <= 0.0:
            continue

        for i in range(n_samples):
            values[i] = X[indices[start + i], feature]

        order = np.argsort(values)
        for pos in range(min_samples_leaf, n_samples - min_samples_leaf + 1):
            left_value = values[order[pos - 1]]
            right_value = values[order[pos]]
            if left_value == right_value:
                continue

            threshold = 0.5 * (left_value + right_value)
            if threshold <= lower[feature] or threshold >= upper[feature]:
                continue

            n_left = pos
            n_right = n_samples - pos
            left_volume = ((threshold - lower[feature]) / width) * volume
            right_volume = volume - left_volume
            left_fraction = n_left / n_total_samples
            right_fraction = n_right / n_total_samples

            left_delta = _bce_delta(
                value,
                left_volume,
                left_fraction,
                l2_reg,
                loss_code,
                bce_fixed_point_iter,
                eps,
            )
            right_delta = _bce_delta(
                value,
                right_volume,
                right_fraction,
                l2_reg,
                loss_code,
                bce_fixed_point_iter,
                eps,
            )
            score = _bce_score(
                value,
                left_volume,
                right_volume,
                left_fraction,
                right_fraction,
                left_delta,
                right_delta,
                l2_reg,
                loss_code,
            )
            if score > best_score:
                best_score = score
                best_feature = feature
                best_threshold = threshold
                best_left_delta = left_delta
                best_right_delta = right_delta

    return (
        best_feature,
        best_threshold,
        best_score,
        best_left_delta,
        best_right_delta,
    )


def _find_random_split_mse_py(
    X: np.ndarray,
    indices: np.ndarray,
    start: int,
    end: int,
    lower: np.ndarray,
    upper: np.ndarray,
    volume: float,
    value: float,
    min_samples_leaf: int,
    l2_reg: float,
    n_total_samples: int,
    random_unit: np.ndarray,
) -> tuple[int, float, float, float, float]:
    n_samples = end - start
    n_features = X.shape[1]
    best_score = -1.0
    best_feature = -1
    best_threshold = 0.0
    best_left_delta = 0.0
    best_right_delta = 0.0

    for feature in range(n_features):
        width = upper[feature] - lower[feature]
        if width <= 0.0:
            continue

        feature_min = np.inf
        feature_max = -np.inf
        for i in range(n_samples):
            feature_value = X[indices[start + i], feature]
            if feature_value < feature_min:
                feature_min = feature_value
            if feature_value > feature_max:
                feature_max = feature_value

        if feature_max <= feature_min:
            continue

        threshold = feature_min + random_unit[feature] * (feature_max - feature_min)
        if threshold <= lower[feature] or threshold >= upper[feature]:
            continue

        n_left = 0
        for i in range(n_samples):
            if X[indices[start + i], feature] <= threshold:
                n_left += 1
        n_right = n_samples - n_left
        if n_left < min_samples_leaf or n_right < min_samples_leaf:
            continue

        left_volume = ((threshold - lower[feature]) / width) * volume
        right_volume = volume - left_volume

        left_fraction = n_left / n_total_samples
        right_fraction = n_right / n_total_samples
        left_linear = value * left_volume + (value - 1.0) * left_fraction
        right_linear = value * right_volume + (value - 1.0) * right_fraction
        left_quad = left_volume + left_fraction + l2_reg
        right_quad = right_volume + right_fraction + l2_reg

        score = left_linear * left_linear / left_quad
        score += right_linear * right_linear / right_quad
        if score > best_score:
            best_score = score
            best_feature = feature
            best_threshold = threshold
            best_left_delta = -left_linear / left_quad
            best_right_delta = -right_linear / right_quad

    return (
        best_feature,
        best_threshold,
        best_score,
        best_left_delta,
        best_right_delta,
    )


def _find_random_split_bce_py(
    X: np.ndarray,
    indices: np.ndarray,
    start: int,
    end: int,
    lower: np.ndarray,
    upper: np.ndarray,
    volume: float,
    value: float,
    min_samples_leaf: int,
    l2_reg: float,
    loss_code: int,
    bce_fixed_point_iter: int,
    eps: float,
    n_total_samples: int,
    random_unit: np.ndarray,
) -> tuple[int, float, float, float, float]:
    n_samples = end - start
    n_features = X.shape[1]
    best_score = -np.inf
    best_feature = -1
    best_threshold = 0.0
    best_left_delta = 0.0
    best_right_delta = 0.0

    for feature in range(n_features):
        width = upper[feature] - lower[feature]
        if width <= 0.0:
            continue

        feature_min = np.inf
        feature_max = -np.inf
        for i in range(n_samples):
            feature_value = X[indices[start + i], feature]
            if feature_value < feature_min:
                feature_min = feature_value
            if feature_value > feature_max:
                feature_max = feature_value

        if feature_max <= feature_min:
            continue

        threshold = feature_min + random_unit[feature] * (feature_max - feature_min)
        if threshold <= lower[feature] or threshold >= upper[feature]:
            continue

        n_left = 0
        for i in range(n_samples):
            if X[indices[start + i], feature] <= threshold:
                n_left += 1
        n_right = n_samples - n_left
        if n_left < min_samples_leaf or n_right < min_samples_leaf:
            continue

        left_volume = ((threshold - lower[feature]) / width) * volume
        right_volume = volume - left_volume
        left_fraction = n_left / n_total_samples
        right_fraction = n_right / n_total_samples
        left_delta = _bce_delta(
            value,
            left_volume,
            left_fraction,
            l2_reg,
            loss_code,
            bce_fixed_point_iter,
            eps,
        )
        right_delta = _bce_delta(
            value,
            right_volume,
            right_fraction,
            l2_reg,
            loss_code,
            bce_fixed_point_iter,
            eps,
        )
        score = _bce_score(
            value,
            left_volume,
            right_volume,
            left_fraction,
            right_fraction,
            left_delta,
            right_delta,
            l2_reg,
            loss_code,
        )
        if score > best_score:
            best_score = score
            best_feature = feature
            best_threshold = threshold
            best_left_delta = left_delta
            best_right_delta = right_delta

    return (
        best_feature,
        best_threshold,
        best_score,
        best_left_delta,
        best_right_delta,
    )


def _predict_values_py(
    X: np.ndarray,
    feature: np.ndarray,
    threshold: np.ndarray,
    value: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    out: np.ndarray,
) -> None:
    for i in range(X.shape[0]):
        node = 0
        while left[node] != -1:
            if X[i, feature[node]] <= threshold[node]:
                node = left[node]
            else:
                node = right[node]
        out[i] = value[node]


def _apply_tree_py(
    X: np.ndarray,
    feature: np.ndarray,
    threshold: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    out: np.ndarray,
) -> None:
    for i in range(X.shape[0]):
        node = 0
        while left[node] != -1:
            if X[i, feature[node]] <= threshold[node]:
                node = left[node]
            else:
                node = right[node]
        out[i] = node


try:
    from numba import njit

    _sigmoid_scalar = njit(cache=True, nogil=True)(_sigmoid_scalar)
    _logit_scalar = njit(cache=True, nogil=True)(_logit_scalar)
    _softplus_scalar = njit(cache=True, nogil=True)(_softplus_scalar)
    _bce_approx_delta = njit(cache=True, nogil=True)(_bce_approx_delta)
    _bce_fixed_point_delta = njit(cache=True, nogil=True)(_bce_fixed_point_delta)
    _bce_delta = njit(cache=True, nogil=True)(_bce_delta)
    _bce_score = njit(cache=True, nogil=True)(_bce_score)
    find_best_split_mse = njit(cache=True, nogil=True)(_find_best_split_mse_py)
    find_best_split_bce = njit(cache=True, nogil=True)(_find_best_split_bce_py)
    find_random_split_mse = njit(cache=True, nogil=True)(_find_random_split_mse_py)
    find_random_split_bce = njit(cache=True, nogil=True)(_find_random_split_bce_py)
    predict_values = njit(cache=True, nogil=True)(_predict_values_py)
    apply_tree = njit(cache=True, nogil=True)(_apply_tree_py)
except Exception:
    find_best_split_mse = _find_best_split_mse_py
    find_best_split_bce = _find_best_split_bce_py
    find_random_split_mse = _find_random_split_mse_py
    find_random_split_bce = _find_random_split_bce_py
    predict_values = _predict_values_py
    apply_tree = _apply_tree_py
