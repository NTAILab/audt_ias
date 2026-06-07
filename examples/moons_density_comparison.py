"""Moons density example with axis-aligned and rotation forests."""
import numpy as np
from sklearn.datasets import make_moons
from audt_ias import UnaryDensityForest


def make_unit_moons(
    n_samples: int = 1200,
    noise: float = 0.035,
    random_state: int = 11,
) -> np.ndarray:
    """Generate a two-moons sample scaled into the unit square.

    Args:
        n_samples: Number of generated samples.
        noise: Standard deviation of Gaussian noise in ``make_moons``.
        random_state: Random seed.

    Returns:
        Feature matrix with shape ``(n_samples, 2)`` in ``[0, 1]^2``.
    """
    X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    lower = X.min(axis=0)
    upper = X.max(axis=0)
    return (X - lower) / (upper - lower)


def fit_forest(rotation: bool | str, random_state: int) -> UnaryDensityForest:
    """Create a compact forest for the moons example.

    Args:
        rotation: Per-tree rotation mode.
        random_state: Random seed.

    Returns:
        Fitted-density estimator configuration.
    """
    return UnaryDensityForest(
        n_estimators=96,
        max_samples=0.8,
        bootstrap=True,
        rotation=rotation,
        max_depth=12,
        min_samples_leaf=4,
        bounds=np.array([[0.0, 0.0], [1.0, 1.0]]),
        bounds_margin=0.0,
        random_state=random_state,
    )


def main() -> None:
    """Compare average log densities on data and uniform background points."""
    rng = np.random.default_rng(11)
    X = make_unit_moons()
    background = rng.uniform(0.0, 1.0, size=(X.shape[0], 2))

    models = {
        "axis_aligned": fit_forest(rotation=False, random_state=0).fit(X),
        "random_rotation": fit_forest(rotation="random", random_state=1).fit(X),
    }

    for name, model in models.items():
        data_log_density = model.score_log_samples(X).mean()
        background_log_density = model.score_log_samples(background).mean()
        print(
            f"{name}: mean_log_density(data)={data_log_density:.3f}, "
            f"mean_log_density(background)={background_log_density:.3f}"
        )


if __name__ == "__main__":
    main()
