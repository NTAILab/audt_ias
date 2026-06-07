"""Loss-specific formulas for unary density trees."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SplitUpdate:
    """Optimal split update for a fixed feature and threshold."""

    score: float
    left_delta: float
    right_delta: float


def sigmoid(value: float) -> float:
    """Compute a numerically stable scalar sigmoid.

    Args:
        value: Logit value.

    Returns:
        Sigmoid value.
    """
    if value >= 0.0:
        return 1.0 / (1.0 + np.exp(-value))
    exp_value = np.exp(value)
    return exp_value / (1.0 + exp_value)


def sigmoid_array(value: np.ndarray) -> np.ndarray:
    """Compute a numerically stable vectorized sigmoid.

    Args:
        value: Logit values.

    Returns:
        Sigmoid values.
    """
    positive = value >= 0.0
    out = np.empty_like(value, dtype=np.float64)
    out[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    out[~positive] = exp_value / (1.0 + exp_value)
    return out


def logit(probability: float, eps: float = 1e-12) -> float:
    """Compute a clipped scalar logit.

    Args:
        probability: Probability value.
        eps: Clipping value.

    Returns:
        Logit value.
    """
    clipped = min(max(probability, eps), 1.0 - eps)
    return float(np.log(clipped / (1.0 - clipped)))


def softplus(value: float) -> float:
    """Compute a numerically stable scalar softplus.

    Args:
        value: Input value.

    Returns:
        Softplus value.
    """
    if value > 0.0:
        return float(value + np.log1p(np.exp(-value)))
    return float(np.log1p(np.exp(value)))


class MSELoss:
    """Mean squared error loss used by the unary classification tree.

    The formulas correspond to the exact second-order objective for
    ``l(q, y) = (q - y) ** 2``.
    """

    name = "mse"
    initial_value = 0.5

    def split_update(
        self,
        value: float,
        left_volume: float,
        right_volume: float,
        n_left: int,
        n_right: int,
        n_total_samples: int,
        l2_reg: float,
    ) -> SplitUpdate:
        """Compute the optimal leaf deltas for a proposed split.

        Args:
            value: Current prediction value in the parent node.
            left_volume: Absolute unit-cube volume assigned to the left child.
            right_volume: Absolute unit-cube volume assigned to the right child.
            n_left: Number of training points assigned to the left child.
            n_right: Number of training points assigned to the right child.
            n_total_samples: Number of training points in the whole fitted
                sample. Empirical mass is global, not conditional on a node.
            l2_reg: L2 regularization coefficient.

        Returns:
            Split update with maximized quadratic gain and child deltas.
        """
        left_fraction = n_left / n_total_samples
        right_fraction = n_right / n_total_samples

        left_linear = value * left_volume + (value - 1.0) * left_fraction
        right_linear = value * right_volume + (value - 1.0) * right_fraction

        left_quad = left_volume + left_fraction + l2_reg
        right_quad = right_volume + right_fraction + l2_reg

        left_delta = -left_linear / left_quad
        right_delta = -right_linear / right_quad
        score = left_linear * left_linear / left_quad
        score += right_linear * right_linear / right_quad

        return SplitUpdate(score=score, left_delta=left_delta, right_delta=right_delta)

    def to_probability(self, value: np.ndarray) -> np.ndarray:
        """Convert stored values to classifier probabilities.

        Args:
            value: Stored tree values.

        Returns:
            Classifier probabilities.
        """
        return value


class BCELogitLoss:
    """Binary cross-entropy loss on logits.

    Args:
        solver: Optimization variant. Supported values are ``"approx"``,
            ``"closed_form"``, and ``"fixed_point"``.
        fixed_point_iter: Number of fixed-point iterations for the
            regularized exact BCE variant.
        eps: Numerical clipping value.
    """

    name = "bce"
    initial_value = 0.0

    def __init__(
        self,
        solver: str = "approx",
        fixed_point_iter: int = 5,
        eps: float = 1e-12,
    ) -> None:
        self.solver = solver
        self.fixed_point_iter = fixed_point_iter
        self.eps = eps

    def split_update(
        self,
        value: float,
        left_volume: float,
        right_volume: float,
        n_left: int,
        n_right: int,
        n_total_samples: int,
        l2_reg: float,
    ) -> SplitUpdate:
        """Compute the optimal or approximate split update.

        Args:
            value: Current parent logit.
            left_volume: Absolute unit-cube volume assigned to the left child.
            right_volume: Absolute unit-cube volume assigned to the right child.
            n_left: Number of training points assigned to the left child.
            n_right: Number of training points assigned to the right child.
            n_total_samples: Number of training points in the whole fitted
                sample. Empirical mass is global, not conditional on a node.
            l2_reg: L2 regularization coefficient.

        Returns:
            Split update with score and child logit deltas.
        """
        left_fraction = n_left / n_total_samples
        right_fraction = n_right / n_total_samples
        left_delta = self._side_delta(value, left_volume, left_fraction, l2_reg)
        right_delta = self._side_delta(value, right_volume, right_fraction, l2_reg)

        if self.solver == "approx":
            score = self._approx_score(
                value,
                left_volume,
                right_volume,
                left_fraction,
                right_fraction,
                l2_reg,
            )
        else:
            score = -(
                self._exact_side_objective(value, left_delta, left_volume, left_fraction, l2_reg)
                + self._exact_side_objective(
                    value,
                    right_delta,
                    right_volume,
                    right_fraction,
                    l2_reg,
                )
            )
        return SplitUpdate(score=score, left_delta=left_delta, right_delta=right_delta)

    def to_probability(self, value: np.ndarray) -> np.ndarray:
        """Convert stored logits to classifier probabilities.

        Args:
            value: Stored logit values.

        Returns:
            Classifier probabilities.
        """
        return sigmoid_array(value)

    def _side_delta(
        self,
        value: float,
        volume: float,
        fraction: float,
        l2_reg: float,
    ) -> float:
        if self.solver == "approx":
            return self._approx_delta(value, volume, fraction, l2_reg)
        if self.solver == "closed_form" or l2_reg == 0.0:
            probability = fraction / (volume + fraction)
            return logit(probability, self.eps) - value
        return self._fixed_point_delta(value, volume, fraction, l2_reg)

    def _approx_delta(
        self,
        value: float,
        volume: float,
        fraction: float,
        l2_reg: float,
    ) -> float:
        probability = sigmoid(value)
        hessian = probability * (1.0 - probability)
        linear = probability * volume + (probability - 1.0) * fraction
        quadratic = hessian * (volume + fraction) + l2_reg
        return -linear / quadratic

    def _fixed_point_delta(
        self,
        value: float,
        volume: float,
        fraction: float,
        l2_reg: float,
    ) -> float:
        delta = self._approx_delta(value, volume, fraction, l2_reg)
        for _ in range(self.fixed_point_iter):
            probability = sigmoid(value + delta)
            delta = (fraction - probability * (volume + fraction)) / l2_reg
        return delta

    def _approx_score(
        self,
        value: float,
        left_volume: float,
        right_volume: float,
        left_fraction: float,
        right_fraction: float,
        l2_reg: float,
    ) -> float:
        probability = sigmoid(value)
        hessian = probability * (1.0 - probability)

        left_linear = probability * left_volume + (probability - 1.0) * left_fraction
        right_linear = probability * right_volume + (probability - 1.0) * right_fraction
        left_quadratic = hessian * (left_volume + left_fraction) + l2_reg
        right_quadratic = hessian * (right_volume + right_fraction) + l2_reg

        score = left_linear * left_linear / left_quadratic
        score += right_linear * right_linear / right_quadratic
        return score

    @staticmethod
    def _exact_side_objective(
        value: float,
        delta: float,
        volume: float,
        fraction: float,
        l2_reg: float,
    ) -> float:
        child_value = value + delta
        objective = volume * softplus(child_value)
        objective += fraction * softplus(-child_value)
        objective += 0.5 * l2_reg * delta * delta
        return objective


def make_loss(
    loss: str | MSELoss | BCELogitLoss,
    bce_solver: str = "approx",
    bce_fixed_point_iter: int = 5,
    eps: float = 1e-12,
) -> MSELoss | BCELogitLoss:
    """Create a loss object by name.

    Args:
        loss: Loss name or an already constructed loss object.
        bce_solver: BCE solver variant.
        bce_fixed_point_iter: Number of BCE fixed-point iterations.
        eps: Numerical clipping value.

    Returns:
        Loss object used by the tree builder.

    Raises:
        ValueError: If the requested loss is not supported.
    """
    if isinstance(loss, (MSELoss, BCELogitLoss)):
        return loss
    if loss == "mse":
        return MSELoss()
    if loss == "bce":
        return BCELogitLoss(
            solver=bce_solver,
            fixed_point_iter=bce_fixed_point_iter,
            eps=eps,
        )
    raise ValueError(f"Unsupported loss={loss!r}. Use 'mse' or 'bce'.")


def density_from_probability(probability: np.ndarray, eps: float) -> np.ndarray:
    """Convert unary classifier probabilities to unit-cube densities.

    Args:
        probability: Estimated probability of a point belonging to data.
        eps: Numerical clipping value.

    Returns:
        Estimated density in normalized coordinates.
    """
    clipped = np.clip(probability, eps, 1.0 - eps)
    return clipped / (1.0 - clipped)
