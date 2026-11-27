
"""Generate validated predictions for the official MP197 holdout dataset."""

from pathlib import Path
import json

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from src.mp197.data_cleaning import normalize_categories
from src.mp197.preprocessing import FEATURE_COLUMNS


# Project paths
ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "data" / "raw" / "holdout_features.csv"
OUTPUT_PATH = ROOT / "reports" / "metrics" / "holdout_predictions.csv"
REPORT_PATH = ROOT / "reports" / "metrics" / "holdout_prediction_report.json"

# MLflow configuration
TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "MP197MetroCrowdRegressor"
MODEL_URI = f"models:/{MODEL_NAME}@champion"

TARGET = "predicted_entries_per_hour"
EXPECTED_ROWS = 250


def predict_holdout():
    """Load the champion model and predict all holdout records."""

    # 1. Load the official holdout dataset.
    df = pd.read_csv(INPUT_PATH)

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} records, found {len(df)}."
        )

    if "entries_per_hour" in df.columns:
        raise ValueError(
            "Holdout dataset must not contain the target column."
        )

    # 2. Validate required columns.
    required_columns = ["record_id", "date"] + FEATURE_COLUMNS
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df[required_columns].isnull().any().any():
        raise ValueError("Missing values found in holdout dataset.")

    if df["record_id"].duplicated().any():
        raise ValueError("Duplicate record_id values found.")

    # 3. Normalize categorical values using training rules.
    cleaned = normalize_categories(df)

    # 4. Select only the eight model input features.
    X = cleaned[FEATURE_COLUMNS].copy()

    if list(X.columns) != FEATURE_COLUMNS:
        raise ValueError("Unexpected model feature order.")

    # 5. Load the registered champion model.
    mlflow.set_tracking_uri(TRACKING_URI)

    model = mlflow.sklearn.load_model(MODEL_URI)

    # 6. Generate raw predictions.
    raw_predictions = np.asarray(
        model.predict(X),
        dtype=float
    )

    if len(raw_predictions) != len(df):
        raise ValueError(
            "Prediction count does not match holdout row count."
        )

    if not np.isfinite(raw_predictions).all():
        raise ValueError("Non-finite predictions detected.")

    # 7. Apply a non-negative output safeguard.
    negative_prediction_count = int(
        (raw_predictions < 0).sum()
    )

    predictions = np.maximum(raw_predictions, 0.0)

    # 8. Prepare the final prediction file.
    output = pd.DataFrame({
        "record_id": df["record_id"],
        "date": df["date"],
        TARGET: predictions
    })

    if len(output) != EXPECTED_ROWS:
        raise ValueError("Final output does not contain 250 records.")

    if output["record_id"].duplicated().any():
        raise ValueError("Duplicate record IDs in prediction output.")

    if output[TARGET].isna().any():
        raise ValueError("Missing predictions in final output.")

    if (output[TARGET] < 0).any():
        raise ValueError("Negative predictions remain after correction.")

    # 9. Save prediction CSV.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(OUTPUT_PATH, index=False)

    # 10. Generate a detailed prediction report.
    report = {
        "dataset": "holdout_features.csv",
        "model_name": MODEL_NAME,
        "model_alias": "champion",
        "model_uri": MODEL_URI,
        "records_received": int(len(df)),
        "predictions_generated": int(len(predictions)),
        "negative_raw_predictions": negative_prediction_count,
        "postprocessing": "Negative predictions clipped to zero",
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
        "prediction_mean": float(predictions.mean()),
        "prediction_median": float(np.median(predictions)),
        "output_file": str(OUTPUT_PATH.relative_to(ROOT)),
        "validation": {
            "expected_records": EXPECTED_ROWS,
            "actual_records": int(len(output)),
            "missing_predictions": int(output[TARGET].isna().sum()),
            "duplicate_record_ids": int(
                output["record_id"].duplicated().sum()
            ),
            "negative_final_predictions": int(
                (output[TARGET] < 0).sum()
            ),
            "passed": True
        }
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    # 11. Display completion summary.
    print("\n" + "=" * 55)
    print("MP197 HOLDOUT PREDICTION COMPLETED")
    print("=" * 55)
    print(json.dumps(report, indent=4))
    print("\nFirst five predictions:")
    print(output.head().to_string(index=False))
    print(f"\nCSV saved: {OUTPUT_PATH}")
    print(f"Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    predict_holdout()
