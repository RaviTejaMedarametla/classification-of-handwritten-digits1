"""Deterministic baseline evaluation for KNN and Random Forest classifiers."""

from sklearn.metrics import accuracy_score

from research.config import SystemConfig, TrainingConfig
from research.core.data.dataset import load_dataset, split_and_normalize
from compression.classical import build_model
from research.core.utils.reproducibility import set_deterministic


def evaluate_model(model_name: str):
    sys_cfg = SystemConfig(seed=40, sample_size=6000, dataset="digits")
    tr_cfg = TrainingConfig(model_name=model_name, rf_random_state=sys_cfg.seed)
    set_deterministic(sys_cfg.seed)
    x, y, _ = load_dataset(sys_cfg)
    x_train, x_test, y_train, y_test = split_and_normalize(x, y, sys_cfg)
    model = build_model(tr_cfg)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return model, accuracy_score(y_test, pred)


def main():
    knn_model, knn_acc = evaluate_model("knn")
    print("KNN baseline")
    print(f"estimator: {knn_model}")
    print(f"accuracy: {knn_acc}")

    rf_model, rf_acc = evaluate_model("rf")
    print("\nRandom Forest baseline")
    print(f"estimator: {rf_model}")
    print(f"accuracy: {rf_acc}")


if __name__ == "__main__":
    main()
