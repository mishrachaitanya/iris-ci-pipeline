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


def train_model_with_epochs(X_train, y_train, X_test, y_test, model_path="model.joblib", metrics_path="metrics.csv", epochs=5):
    metrics = []

    print("Training started...")
    for epoch in range(1, epochs + 1):
        clf = RandomForestClassifier(n_estimators=10 * epoch, random_state=epoch)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_proba)
        metrics.append({"epoch": epoch, "accuracy": acc, "loss": loss})

        print(f"Epoch {epoch}: Accuracy={acc:.4f}, Loss={loss:.4f}")

        # Save final model
        if epoch == epochs:
            joblib.dump(clf, model_path)
            print(f"Model saved to {model_path}")

    # Save metrics.csv
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "accuracy", "loss"])
        writer.writeheader()
        writer.writerows(metrics)

    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="model.joblib")
    parser.add_argument("--metrics-path", default="metrics.csv")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_data()
    train_model_with_epochs(X_train, y_train, X_test, y_test,
                            model_path=args.model_path,
                            metrics_path=args.metrics_path)

