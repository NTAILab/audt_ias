"""Rotation forest anomaly-scoring example."""
import numpy as np
from sklearn.metrics import roc_auc_score
from audt_ias import UnaryDensityForest


def main() -> None:
    """Fit a random-rotation forest and evaluate rarity scores."""
    rng = np.random.default_rng(2)
    normal_train = rng.normal(loc=0.0, scale=0.8, size=(600, 4))
    normal_test = rng.normal(loc=0.0, scale=0.8, size=(250, 4))
    anomalies = rng.normal(loc=3.0, scale=0.7, size=(80, 4))

    model = UnaryDensityForest(
        n_estimators=64,
        max_samples=0.8,
        bootstrap=True,
        rotation="random",
        rotation_bounds="data_aabb",
        max_depth=9,
        min_samples_leaf=6,
        leaf_value_mode="refit",
        density_aggregation="mean_log_density",
        n_jobs=1,
        random_state=2,
    ).fit(normal_train)

    X_eval = np.vstack([normal_test, anomalies])
    y_eval = np.array([0] * normal_test.shape[0] + [1] * anomalies.shape[0])
    anomaly_score = -model.score_log_samples(X_eval)
    auc = roc_auc_score(y_eval, anomaly_score)

    print(f"AUROC: {auc:.3f}")
    print(f"Mean normal score: {anomaly_score[y_eval == 0].mean():.3f}")
    print(f"Mean anomaly score: {anomaly_score[y_eval == 1].mean():.3f}")


if __name__ == "__main__":
    main()
