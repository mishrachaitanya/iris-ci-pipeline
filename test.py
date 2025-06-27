# test.py

import joblib
from train import load_data, train_model, save_local, evaluate_model
from sklearn.metrics import accuracy_score
from pathlib import Path


def test_training_and_saving(tmp_path):
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train, n_estimators=10)
    preds = model.predict(X_test)
    assert accuracy_score(y_test, preds) > 0.85, "Model too weak"

    model_file = tmp_path / "iris_model.joblib"
    saved = save_local(model, model_file)
    assert saved.exists(), "Model file not saved"


def test_saved_model_prediction(tmp_path):
    # Simulate training and saving
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train, n_estimators=10)
    model_path = tmp_path / "iris_model.joblib"
    save_local(model, model_path)

    # Load model back
    loaded_model = joblib.load(model_path)
    pred = loaded_model.predict([X_test.iloc[0]])[0]
    assert pred in [0, 1, 2], "Invalid prediction from loaded model"
