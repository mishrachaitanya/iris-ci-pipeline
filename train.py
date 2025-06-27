# train.py

import argparse
import joblib
import csv
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss


def load_data():
    X, y = load_iris(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train, n_estimators=100, max_depth=4):
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba)
    return {"accuracy": acc, "loss": loss}


def save_local(model, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def write_metrics(metrics: dict, out_path="metrics.csv"):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "accuracy", "loss"])
        writer.writeheader()
        writer.writerow({"epoch": 1, "accuracy": metrics["accuracy"], "loss": metrics["loss"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="model.joblib")
    parser.add_argument("--metrics-path", default="metrics.csv")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    save_local(model, args.model_path)
    write_metrics(metrics, args.metrics_path)

    print(f"Model saved to {args.model_path}")
    print(f"Metrics written to {args.metrics_path}")

