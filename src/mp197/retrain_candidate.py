from pathlib import Path
import json
import hashlib
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.mp197.data_cleaning import (
    clean_training_data,
    clean_later_data,
)
from src.mp197.preprocessing import (
    build_preprocessor,
    FEATURE_COLUMNS,
)


TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "MP197MetroCrowdRegressor"
TARGET = "entries_per_hour"

TRAIN_PATH = "data/raw/train.csv"
LATER_PATH = "data/raw/later.csv"

def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()

def calculate_metrics(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def main():
    mlflow.set_tracking_uri(TRACKING_URI)

    # Load and clean both datasets.
    original_train, _ = clean_training_data(TRAIN_PATH)
    later = clean_later_data(LATER_PATH)

    later["date"] = pd.to_datetime(later["date"], errors="raise")
    later = later.sort_values(["date", "record_id"]).reset_index(drop=True)

    # Select a date boundary near 70% of later records.
    split_position = int(len(later) * 0.70)
    cutoff_date = later.loc[split_position, "date"]

    earlier_later = later[later["date"] < cutoff_date].copy()
    future_validation = later[later["date"] >= cutoff_date].copy()

    if earlier_later.empty or future_validation.empty:
        raise ValueError("Chronological split produced an empty partition.")

    # Keep only model features and target.
    training_columns = FEATURE_COLUMNS + [TARGET]

    candidate_training = pd.concat(
        [
            original_train[training_columns],
            earlier_later[training_columns],
        ],
        ignore_index=True,
    )

    X_train = candidate_training[FEATURE_COLUMNS]
    y_train = candidate_training[TARGET]

    X_valid = future_validation[FEATURE_COLUMNS]
    y_valid = future_validation[TARGET]

    # Load the existing champion before registering the candidate.
    champion = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}@champion"
    )

    champion_predictions = champion.predict(X_valid)
    champion_metrics = calculate_metrics(
        y_valid,
        champion_predictions,
    )

    # Train the new candidate.
    candidate = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "regressor",
                GradientBoostingRegressor(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42,
                ),
            ),
        ]
    )

    candidate.fit(X_train, y_train)

    candidate_predictions = candidate.predict(X_valid)
    candidate_metrics = calculate_metrics(
        y_valid,
        candidate_predictions,
    )

    # Register candidate, but do not change the champion alias.
    with mlflow.start_run(run_name="gradient_boosting_retrained_candidate") as run:

        mlflow.log_param("training_source", "original_train_plus_earlier_later")
        mlflow.log_param("training_rows", len(candidate_training))
        mlflow.log_param(
            "data_sha256",
            calculate_sha256(TRAIN_PATH)
        )
        mlflow.log_param("validation_rows", len(future_validation))
        mlflow.log_param("validation_start_date", str(cutoff_date.date()))
        mlflow.log_param("validation_end_date", str(future_validation["date"].max().date()))
        mlflow.log_param("candidate_model", "GradientBoostingRegressor")

        mlflow.log_metric("candidate_mae", candidate_metrics["mae"])
        mlflow.log_metric("candidate_rmse", candidate_metrics["rmse"])
        mlflow.log_metric("candidate_r2", candidate_metrics["r2"])

        mlflow.log_metric("champion_mae_same_validation", champion_metrics["mae"])
        mlflow.log_metric("champion_rmse_same_validation", champion_metrics["rmse"])
        mlflow.log_metric("champion_r2_same_validation", champion_metrics["r2"])

        signature = infer_signature(
            X_train,
            candidate.predict(X_train.head(5)),
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=candidate,
            name="candidate_model",
            signature=signature,
            input_example=X_train.head(3),
            registered_model_name=MODEL_NAME,
            serialization_format="cloudpickle",
        )

        candidate_version = model_info.registered_model_version
        run_id = run.info.run_id

    # Calculate improvement: positive means candidate has lower MAE.
    improvement_percent = (
        (champion_metrics["mae"] - candidate_metrics["mae"])
        / champion_metrics["mae"]
        * 100
    )

    report = {
        "training_rows": len(candidate_training),
        "validation_rows": len(future_validation),
        "validation_start_date": str(cutoff_date.date()),
        "validation_end_date": str(future_validation["date"].max().date()),
        "zero_target_rows_in_candidate_training": int(
            (earlier_later[TARGET] == 0).sum()
        ),
        "champion_metrics": champion_metrics,
        "candidate_metrics": candidate_metrics,
        "candidate_mae_improvement_percent": float(improvement_percent),
        "candidate_model_version": str(candidate_version),
        "mlflow_run_id": run_id,
        "champion_alias_changed": False,
    }

    output_dir = Path("reports/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(
        output_dir / "candidate_validation.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=4)

    predictions = pd.DataFrame({
        "record_id": future_validation["record_id"],
        "date": future_validation["date"],
        "actual": y_valid,
        "champion_prediction": champion_predictions,
        "candidate_prediction": candidate_predictions,
    })

    predictions.to_csv(
        output_dir / "candidate_predictions.csv",
        index=False,
    )

    print("\nCandidate validation completed.")
    print(f"Training rows: {len(candidate_training)}")
    print(f"Validation rows: {len(future_validation)}")
    print(f"Validation dates: {cutoff_date.date()} onward")

    print("\nExisting champion:")
    print(json.dumps(champion_metrics, indent=4))

    print("\nRetrained candidate:")
    print(json.dumps(candidate_metrics, indent=4))

    print(f"\nCandidate MAE improvement: {improvement_percent:.2f}%")
    print(f"Registered candidate version: {candidate_version}")
    print("Champion alias remains unchanged.")
    print("Reports saved in reports/metrics/")


if __name__ == "__main__":
    main()