from pathlib import Path
import json

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.mp197.data_cleaning import clean_later_data
from src.mp197.preprocessing import FEATURE_COLUMNS


TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "MP197_Metro_Crowd_Prediction"
MODEL_NAME = "MP197MetroCrowdRegressor"
TARGET = "entries_per_hour"

LATER_PATH = "data/raw/later.csv"


def evaluate_champion():

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Clean category spellings without changing the original CSV.
    df = clean_later_data(LATER_PATH)

    required_columns = FEATURE_COLUMNS + [TARGET]
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df[required_columns].isnull().any().any():
        raise ValueError("Missing values found in later dataset.")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    # Load the already-selected champion model.
    model_uri = f"models:/{MODEL_NAME}@champion"
    model = mlflow.sklearn.load_model(model_uri)

    predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)
    rmse = mean_squared_error(y, predictions) ** 0.5
    r2 = r2_score(y, predictions)

    # Record temporal evaluation in MLflow.
    with mlflow.start_run(run_name="later_temporal_evaluation") as run:

        mlflow.log_param("evaluation_dataset", "later.csv")
        mlflow.log_param("evaluation_rows", len(df))
        mlflow.log_param("model_alias", "champion")

        mlflow.log_metric("later_mae", float(mae))
        mlflow.log_metric("later_rmse", float(rmse))
        mlflow.log_metric("later_r2", float(r2))

        run_id = run.info.run_id

    # Save evaluation metrics.
    output_dir = Path("reports/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "dataset": "later.csv",
        "rows_evaluated": len(df),
        "model_name": MODEL_NAME,
        "model_alias": "champion",
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mlflow_run_id": run_id,
    }

    with open(
        output_dir / "later_evaluation.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=4)

    # Save row-level predictions for analysis.
    prediction_output = pd.DataFrame({
        "record_id": df["record_id"],
        "date": df["date"],
        "actual_entries_per_hour": y,
        "predicted_entries_per_hour": predictions,
    })

    prediction_output["absolute_error"] = (
        prediction_output["actual_entries_per_hour"]
        - prediction_output["predicted_entries_per_hour"]
    ).abs()

    prediction_output.to_csv(
        output_dir / "later_predictions.csv",
        index=False,
    )

    print("\nTemporal evaluation completed.")
    print(f"Rows evaluated: {len(df)}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"MLflow run ID: {run_id}")
    print("Reports saved in reports/metrics/")


if __name__ == "__main__":
    evaluate_champion()