from pathlib import Path
import json

DRIFT_PATH = Path("reports/metrics/drift_report.json")
EVALUATION_PATH = Path("reports/metrics/later_evaluation.json")
TRAINING_PATH = Path("reports/metrics/training_results.json")
OUTPUT_PATH = Path("reports/metrics/retraining_decision.json")


def make_retraining_decision():
    with open(DRIFT_PATH, encoding="utf-8") as file:
        drift = json.load(file)

    with open(EVALUATION_PATH, encoding="utf-8") as file:
        evaluation = json.load(file)

    with open(TRAINING_PATH, encoding="utf-8") as file:
        training = json.load(file)

    original_mae = float(training["best_model"]["mae"])
    later_mae = float(evaluation["mae"])

    if original_mae <= 0:
        raise ValueError("Original MAE must be positive.")

    mae_ratio = later_mae / original_mae

    high_drift = drift["high_drift_features"]
    moderate_drift = drift["moderate_drift_features"]

    performance_degraded = mae_ratio >= 1.5
    significant_drift = len(high_drift) > 0

    retraining_recommended = (
        performance_degraded and significant_drift
    )

    decision = {
        "original_test_mae": original_mae,
        "later_dataset_mae": later_mae,
        "mae_increase_ratio": mae_ratio,
        "high_drift_features": high_drift,
        "moderate_drift_features": moderate_drift,
        "performance_degraded": performance_degraded,
        "significant_feature_drift": significant_drift,
        "retraining_recommended": retraining_recommended,
        "decision": (
            "RETRAINING_RECOMMENDED"
            if retraining_recommended
            else "CONTINUE_MONITORING"
        ),
        "reason": (
            "Later-set performance degraded substantially and "
            "high feature drift was detected. Validate a retrained "
            "candidate before promoting it."
            if retraining_recommended
            else "Current evidence does not meet both retraining criteria."
        )
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(decision, file, indent=4)

    print(json.dumps(decision, indent=4))


if __name__ == "__main__":
    make_retraining_decision()