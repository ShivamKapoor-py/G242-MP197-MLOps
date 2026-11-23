from pathlib import Path
import hashlib
import json

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.mp197.preprocessing import build_preprocessor, FEATURE_COLUMNS


# Project configuration
DATA_PATH = Path("data/processed/train_clean.csv")
RAW_DATA_PATH = Path("data/raw/train.csv")
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "MP197_Metro_Crowd_Prediction"
REGISTERED_MODEL_NAME = "MP197MetroCrowdRegressor"
TARGET = "entries_per_hour"
RANDOM_STATE = 42


def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {DATA_PATH}")

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Original dataset not found: {RAW_DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURE_COLUMNS + [TARGET]
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    if df[required_columns].isnull().any().any():
        raise ValueError("Missing values found in training data.")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    data_sha256 = calculate_sha256(RAW_DATA_PATH)

    return X, y, data_sha256


def get_models():
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }


def train_and_track():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y, data_sha256 = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    results = []

    for model_name, estimator in get_models().items():

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("regressor", estimator),
            ]
        )

        with mlflow.start_run(run_name=model_name) as run:

            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)
            rmse = mean_squared_error(y_test, predictions) ** 0.5
            r2 = r2_score(y_test, predictions)

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("test_size", 0.20)
            mlflow.log_param("data_sha256", data_sha256)

            mlflow.log_metric("mae", float(mae))
            mlflow.log_metric("rmse", float(rmse))
            mlflow.log_metric("r2", float(r2))

            signature = infer_signature(
                X_train,
                pipeline.predict(X_train),
            )

            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                signature=signature,
                input_example=X_train.head(3),
                registered_model_name=REGISTERED_MODEL_NAME,
                serialization_format="cloudpickle",
            )

            version = model_info.registered_model_version

            results.append(
                {
                    "model_name": model_name,
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "r2": float(r2),
                    "run_id": run.info.run_id,
                    "model_version": str(version),
                }
            )

            print(
                f"{model_name}: MAE={mae:.4f}, "
                f"RMSE={rmse:.4f}, R2={r2:.4f}"
            )

    # Select the model with the lowest MAE.
    best_model = min(results, key=lambda item: item["mae"])

    client = MlflowClient()

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        "champion",
        best_model["model_version"],
    )

    Path("reports/metrics").mkdir(parents=True, exist_ok=True)

    report_path = Path("reports/metrics/training_results.json")

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "data_sha256": data_sha256,
                "models": results,
                "best_model": best_model,
            },
            file,
            indent=4,
        )

    print("\nTraining completed.")
    print(f"Best model: {best_model['model_name']}")
    print(f"Best MAE: {best_model['mae']:.4f}")
    print(f"Champion version: {best_model['model_version']}")
    print(f"Results saved to: {report_path}")


if __name__ == "__main__":
    train_and_track()