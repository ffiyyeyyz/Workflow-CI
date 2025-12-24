import mlflow
import mlflow.sklearn
import pandas as pd
import sys
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

if __name__ == "__main__":
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else None
    min_samples_split = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    min_samples_leaf = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    print("Starting RF training with params:")
    print(
        f"n_estimators={n_estimators}, "
        f"max_depth={max_depth}, "
        f"min_samples_split={min_samples_split}, "
        f"min_samples_leaf={min_samples_leaf}"
    )

    data_path = "hotel_bookings_preprocessing.csv"
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found")
        sys.exit(1)

    df = pd.read_csv(data_path)

    X = df.drop(columns=["is_canceled"])
    y = df["is_canceled"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    with mlflow.start_run():

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "model")

        print(
            f"Training finished | "
            f"Acc={acc:.4f} | "
            f"Prec={prec:.4f} | "
            f"Recall={rec:.4f} | "
            f"F1={f1:.4f}"
        )
