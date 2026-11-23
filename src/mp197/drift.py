from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.mp197.data_cleaning import (
    clean_training_data,
    clean_later_data,
)
from src.mp197.preprocessing import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


TRAIN_PATH = "data/raw/train.csv"
LATER_PATH = "data/raw/later.csv"

REPORT_PATH = Path("reports/metrics/drift_report.json")


def calculate_psi(expected, actual, bins=10):
    """Calculate PSI using training-data quantile bins."""

    expected = pd.Series(expected).dropna().astype(float)
    actual = pd.Series(actual).dropna().astype(float)

    quantiles = np.linspace(0, 1, bins + 1)[1:-1]
    internal_edges = np.unique(expected.quantile(quantiles).to_numpy())

    edges = np.concatenate(([-np.inf], internal_edges, [np.inf]))

    expected_counts = np.histogram(expected, bins=edges)[0]
    actual_counts = np.histogram(actual, bins=edges)[0]

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Small smoothing value prevents division by zero.
    expected_pct = np.clip(expected_pct, 0.0001, None)
    actual_pct = np.clip(actual_pct, 0.0001, None)

    return float(
        np.sum(
            (actual_pct - expected_pct)
            * np.log(actual_pct / expected_pct)
        )
    )


def calculate_categorical_psi(expected, actual):
    """Calculate PSI for categorical feature distributions."""

    expected = expected.astype(str)
    actual = actual.astype(str)

    categories = sorted(set(expected) | set(actual))

    expected_pct = (
        expected.value_counts(normalize=True)
        .reindex(categories, fill_value=0)
    )

    actual_pct = (
        actual.value_counts(normalize=True)
        .reindex(categories, fill_value=0)
    )

    expected_pct = expected_pct.clip(lower=0.0001)
    actual_pct = actual_pct.clip(lower=0.0001)

    return float(
        ((actual_pct - expected_pct)
         * np.log(actual_pct / expected_pct)).sum()
    )


def interpret_psi(value):
    if value < 0.10:
        return "low"
    if value < 0.20:
        return "moderate"
    return "high"


def run_drift_analysis():

    train_df, _ = clean_training_data(TRAIN_PATH)
    later_df = clean_later_data(LATER_PATH)

    results = {}

    # Numeric feature drift.
    for column in NUMERIC_FEATURES:

        train_values = train_df[column].dropna()
        later_values = later_df[column].dropna()

        psi = calculate_psi(train_values, later_values)
        ks_stat, ks_pvalue = ks_2samp(
            train_values,
            later_values,
        )

        results[column] = {
            "feature_type": "numeric",
            "train_mean": float(train_values.mean()),
            "later_mean": float(later_values.mean()),
            "train_min": float(train_values.min()),
            "train_max": float(train_values.max()),
            "later_min": float(later_values.min()),
            "later_max": float(later_values.max()),
            "psi": psi,
            "psi_severity": interpret_psi(psi),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "ks_significant_at_0_05": bool(ks_pvalue < 0.05),
        }

    # Categorical feature drift.
    for column in CATEGORICAL_FEATURES:

        psi = calculate_categorical_psi(
            train_df[column],
            later_df[column],
        )

        results[column] = {
            "feature_type": "categorical",
            "train_distribution": (
                train_df[column]
                .value_counts(normalize=True)
                .to_dict()
            ),
            "later_distribution": (
                later_df[column]
                .value_counts(normalize=True)
                .to_dict()
            ),
            "psi": psi,
            "psi_severity": interpret_psi(psi),
        }

    high_drift_features = [
        name
        for name, values in results.items()
        if values["psi_severity"] == "high"
    ]

    moderate_drift_features = [
        name
        for name, values in results.items()
        if values["psi_severity"] == "moderate"
    ]

    report = {
        "reference_dataset": "cleaned train.csv",
        "comparison_dataset": "later.csv",
        "training_rows": len(train_df),
        "later_rows": len(later_df),
        "features": results,
        "high_drift_features": high_drift_features,
        "moderate_drift_features": moderate_drift_features,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nFeature drift analysis completed.")
    print("\nFeature-wise PSI results:")

    for name, values in results.items():
        print(
            f"{name}: PSI={values['psi']:.4f}, "
            f"severity={values['psi_severity']}"
        )

    print("\nHigh drift:", high_drift_features)
    print("Moderate drift:", moderate_drift_features)
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    run_drift_analysis()