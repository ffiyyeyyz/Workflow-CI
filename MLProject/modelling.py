import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
import argparse
import sys
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def parse_args():
    parser = argparse.ArgumentParser(description="Train Random Forest Model")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    # 1. Gunakan argparse (Lebih aman daripada sys.argv manual)
    args = parse_args()

    print("Starting RF training with params:")
    print(f"  n_estimators: {args.n_estimators}")
    print(f"  max_depth: {args.max_depth}")
    print(f"  min_samples_split: {args.min_samples_split}")
    print(f"  min_samples_leaf: {args.min_samples_leaf}")

    data_path = "hotel_bookings_preprocessing.csv"
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found")
        sys.exit(1)

    # 2. Load Data
    df = pd.read_csv(data_path)
    
    # Pastikan target column ada
    if "is_canceled" not in df.columns:
        print("ERROR: Column 'is_canceled' not found in dataset")
        sys.exit(1)

    X = df.drop(columns=["is_canceled"])
    y = df["is_canceled"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Mulai MLflow Run
    with mlflow.start_run() as run:
        
        mlflow.autolog()

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }
        
        mlflow.log_metrics(metrics)


