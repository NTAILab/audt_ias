"""Basic AUDT density estimation example."""
import numpy as np
from audt_ias import UnaryDensityTree


def main() -> None:
    """Fit one tree and print density estimates for a few points."""
    rng = np.random.default_rng(1)
    left = rng.normal(loc=(-1.0, 0.0), scale=0.25, size=(250, 2))
    right = rng.normal(loc=(1.0, 0.0), scale=0.35, size=(250, 2))
    X = np.vstack([left, right])

    model = UnaryDensityTree(
        max_depth=7,
        min_samples_leaf=8,
        loss="bce",
        bce_solver="closed_form",
        l2_reg=0.0,
        leaf_value_mode="refit",
        bounds_margin=0.1,
        random_state=1,
    ).fit(X)

    query = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [3.0, 3.0]])
    density = model.score_samples(query)
    leaves = model.apply(query)

    for point, value, leaf in zip(query, density, leaves):
        print(f"x={point.tolist()} density={value:.6f} leaf={int(leaf)}")


if __name__ == "__main__":
    main()
