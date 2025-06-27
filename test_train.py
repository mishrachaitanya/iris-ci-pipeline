# test_train.py
import csv
import joblib
from pathlib import Path

from train import (
    load_data,
    train_model,
    train_model_with_epochs,
    save_local,
)
from sklearn.metrics import accuracy_score


def test_training_and_saving(tmp_path):
    """Train once, save, and make sure accuracy is reasonable."""
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train, n_estimators=10)
    preds = model.predict(X_test)
    assert accuracy_score(y_test, preds) > 0.80, "Model too weak"

    model_file = tmp_path / "iris_model.joblib"
    saved = save_local(model, model_file)
    assert saved.exists(), "Model file not saved"


def test_saved_model_prediction(tmp_path):
    """
    Call the real training routine that writes model.joblib,
    then load that same file and do a single prediction.
    """
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.csv"

    # run the multi-epoch trainer so it produces both artefacts
    X_train, X_test, y_train, y_test = load_data()
    train_model_with_epochs(
        X_train,
        y_train,
        X_test,
        y_test,
        model_path=model_path,
        metrics_path=metrics_path,
        epochs=3,
    )

    # 1️⃣ model exists and can be loaded
    assert model_path.exists(), "model.joblib was not written"
    loaded = joblib.load(model_path)
    pred = loaded.predict([X_test.iloc[0]])[0]
    assert pred in [0, 1, 2], "Invalid prediction from loaded model"

    # 2️⃣ metrics.csv has a header + 3 data rows
    assert metrics_path.exists(), "metrics.csv was not written"
    with open(metrics_path) as f:
        rows = list(csv.reader(f))
    assert len(rows) == 1 + 3, "metrics.csv does not contain 3 epochs"

