from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from ml_system.config import TrainingConfig


def build_model(config: TrainingConfig):
    if config.model_name == "knn":
        return KNeighborsClassifier(
            n_neighbors=config.knn_neighbors,
            weights=config.knn_weights,
            algorithm=config.knn_algorithm,
        )
    if config.model_name == "rf":
        return RandomForestClassifier(
            n_estimators=config.rf_n_estimators,
            max_features=config.rf_max_features,
            class_weight=config.rf_class_weight,
            random_state=40,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model_name={config.model_name}")
