# tuning.py

import tensorflow as tf
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

def load_and_reshape_data():
    """Load the MNIST dataset and reshape the images to be 1D arrays."""
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

    # Use the first 6000 rows of the dataset
    x_train = x_train[:6000]
    y_train = y_train[:6000]

    # Flatten the training data (28x28 pixels into 1D array of 784 elements)
    x_train_flat = x_train.reshape(x_train.shape[0], 28 * 28)

    return x_train_flat, y_train

def split_dataset(x, y):
    """Split the dataset into training and test sets with a 70-30 split."""
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=40)
    return x_train, x_test, y_train, y_test

def normalize_data(x_train, x_test):
    """Normalize the training and test datasets."""
    normalizer = Normalizer()
    x_train_norm = normalizer.fit_transform(x_train)
    x_test_norm = normalizer.transform(x_test)
    return x_train_norm, x_test_norm

def grid_search_knn(x_train, y_train):
    """Perform GridSearchCV for K-Nearest Neighbors classifier."""
    param_grid = {
        'n_neighbors': [3, 4],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'brute']
    }
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x_train, y_train)
    return grid_search

def grid_search_rf(x_train, y_train):
    """Perform GridSearchCV for Random Forest classifier."""
    param_grid = {
        'n_estimators': [300, 500],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    rf = RandomForestClassifier(random_state=40)
    grid_search = GridSearchCV(rf, param_grid, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x_train, y_train)
    return grid_search

def main():
    # Load and reshape the dataset
    x, y = load_and_reshape_data()

    # Split the dataset into training and test sets
    x_train, x_test, y_train, y_test = split_dataset(x, y)

    # Normalize the training and test sets
    x_train_norm, x_test_norm = normalize_data(x_train, x_test)

    # Perform grid search on K-Nearest Neighbors
    knn_grid_search = grid_search_knn(x_train_norm, y_train)
    knn_best_estimator = knn_grid_search.best_estimator_
    print(f"K-nearest neighbours algorithm")
    print(f"best estimator: {knn_best_estimator}")

    # Evaluate the best KNN model on the test set
    knn_predictions = knn_best_estimator.predict(x_test_norm)
    knn_accuracy = accuracy_score(y_test, knn_predictions)
    print(f"accuracy: {knn_accuracy}")

    # Perform grid search on Random Forest
    rf_grid_search = grid_search_rf(x_train_norm, y_train)
    rf_best_estimator = rf_grid_search.best_estimator_
    print(f"\nRandom forest algorithm")
    print(f"best estimator: {rf_best_estimator}")

    # Evaluate the best Random Forest model on the test set
    rf_predictions = rf_best_estimator.predict(x_test_norm)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    print(f"accuracy: {rf_accuracy}")

if __name__ == "__main__":
    main()
