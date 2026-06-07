"""Class-conditional density-ratio classifier example."""
import numpy as np
from audt_ias import UnaryDensityBayesClassifier, UnaryDensityForest


def main() -> None:
    """Fit a Bayes classifier using unary density forests."""
    rng = np.random.default_rng(3)
    X0 = rng.normal(loc=(-1.0, 0.0), scale=0.45, size=(250, 2))
    X1 = rng.normal(loc=(1.0, 0.2), scale=0.45, size=(250, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * X0.shape[0] + [1] * X1.shape[0])

    density_estimator = UnaryDensityForest(
        n_estimators=32,
        max_samples=0.85,
        rotation="random",
        max_depth=7,
        min_samples_leaf=5,
        leaf_value_mode="refit",
        n_jobs=1,
        random_state=3,
    )
    classifier = UnaryDensityBayesClassifier(
        density_estimator=density_estimator,
        margin=0.03,
        use_class_prior=True,
    ).fit(X, y)

    proba = classifier.predict_proba(np.array([[-1.0, 0.0], [1.0, 0.2]]))
    print(np.round(proba, 3))


if __name__ == "__main__":
    main()
