# train.py
import argparse
import csv
import joblib
from pathlib import Path
from typing import Dict, Tuple

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split


# ---------- data ------------------------------------------------------------
def load_data() -> Tuple:
    """Return train / test splits of the Iris dataset."""
    X, y = load_iris(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ---------- single-shot training (used by tests) ----------------------------
def train_model(
    X_train,
    y_train,
    n_estimators: int = 100,
    max_depth: int | None = 4,
):
    """Train one RandomForest (this is what the tests import)."""
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    """Return accuracy & log-loss."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "loss": log_loss(y_test, y_proba),
    }


def save_local(model, path: str | Path) -> Path:
    """Persist the model with joblib and return the Path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


# ---------- multi-epoch demo (CLI entry-point) ------------------------------
def train_model_with_epochs(
    X_train,
    y_train,
    X_test,
    y_test,
    model_path="model.joblib",
    metrics_path="metrics.csv",
    epochs: int = 5,
):
    """Train a fresh forest each epoch just for demonstration/metrics."""
    metrics_rows = []

    print("Training started …")
    for epoch in range(1, epochs + 1):
        clf = RandomForestClassifier(n_estimators=10 * epoch, random_state=epoch)
        clf.fit(X_train, y_train)

        scores = evaluate_model(clf, X_test, y_test)
        metrics_rows.append({"epoch": epoch, **scores})

        print(f"Epoch {epoch}: accuracy={scores['accuracy']:.4f}   loss={scores['loss']:.4f}")

        # keep the last epoch’s model
        if epoch == epochs:
            save_local(clf, model_path)
            print(f"Model saved to {model_path}")

    # write metrics.csv
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "accuracy", "loss"])
        writer.writeheader()
        writer.writerows(metrics_rows)
    print(f"Metrics saved to {metrics_path}")


# ---------- CLI -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="model.joblib")
    parser.add_argument("--metrics-path", default="metrics.csv")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_data()
    train_model_with_epochs(
        X_train,
        y_train,
        X_test,
        y_test,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
    )

