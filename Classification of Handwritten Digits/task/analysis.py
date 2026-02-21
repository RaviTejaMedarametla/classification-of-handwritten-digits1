"""Compatibility entrypoint preserving original project behavior."""

from sklearn.metrics import accuracy_score

from ml_system.config import SystemConfig, TrainingConfig
from ml_system.data.dataset import load_mnist, split_and_normalize
from ml_system.models.classical import build_model
from ml_system.utils.reproducibility import set_deterministic


def evaluate_model(model_name: str):
    sys_cfg = SystemConfig(seed=40, sample_size=6000)
    tr_cfg = TrainingConfig(model_name=model_name)
    set_deterministic(sys_cfg.seed)
    x, y = load_mnist(sys_cfg)
    x_train, x_test, y_train, y_test = split_and_normalize(x, y, sys_cfg)
    model = build_model(tr_cfg)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return model, accuracy_score(y_test, pred)


def main():
    knn_model, knn_acc = evaluate_model("knn")
    print("K-nearest neighbours algorithm")
    print(f"best estimator: {knn_model}")
    print(f"accuracy: {knn_acc}")

    rf_model, rf_acc = evaluate_model("rf")
    print("\nRandom forest algorithm")
    print(f"best estimator: {rf_model}")
    print(f"accuracy: {rf_acc}")


if __name__ == "__main__":
    main()
