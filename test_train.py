#Comment 
#ANother comment
import csv
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from train import (
    load_data,
    train_model,
    train_model_with_epochs,
    save_local
)

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score


def test_saved_model_prediction(tmp_path):
    """Trains model, checks prediction and metrics.csv"""
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.csv"

    X_train, X_test, y_train, y_test = load_data()
    train_model_with_epochs(
        X_train, y_train, X_test, y_test,
        model_path=model_path,
        metrics_path=metrics_path,
        epochs=3,
    )

    loaded = joblib.load(model_path)
    row = X_test.iloc[[0]]  # keep column names
    pred = loaded.predict(row)[0]
    assert pred in [0, 1, 2]

    with open(metrics_path) as f:
        rows = list(csv.reader(f))
    assert len(rows) == 4  # 1 header + 3 rows


def test_random_input_on_saved_model():
    """
    Loads model.joblib saved by train.py (outside test), 
    creates valid-shaped random input, predicts, and checks output.
    """
    model_path = Path("model.joblib")
    assert model_path.exists(), "You must run train.py first to generate model.joblib"

    model = joblib.load(model_path)

    # load feature names from original iris dataset
    iris = load_iris()
    feature_names = iris.feature_names
    n_features = len(feature_names)

    # generate a single random input sample within valid range
    feature_min = iris.data.min(axis=0)
    feature_max = iris.data.max(axis=0)
    random_sample = np.random.uniform(low=feature_min, high=feature_max, size=(1, n_features))
    df_sample = pd.DataFrame(random_sample, columns=feature_names)

    pred = model.predict(df_sample)[0]
    assert pred in [0, 1, 2], f"Prediction {pred} is invalid"

