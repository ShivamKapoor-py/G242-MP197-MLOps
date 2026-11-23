```python
"""Data cleaning pipeline for G242 MP197."""

from pathlib import Path
import json

import pandas as pd


# Target column representing the hourly passenger entry count.
TARGET = "entries_per_hour"

# Dates and temperature value associated with the identified sensor issue.
STUCK_START = pd.Timestamp("2026-05-07")
STUCK_END = pd.Timestamp("2026-05-28")
STUCK_VALUE = 42.7

# Standardize inconsistent day-type labels into common categories.
DAY_TYPE_MAPPING = {
    "weekday": "weekday",
    "weekend": "weekend",
    "holiday": "holiday",
    "hol.": "holiday",
    "wee.": "weekend",
}


def normalize_categories(df):
    """Standardize categorical values."""
    df = df.copy()

    # Convert day-type values to lowercase, remove whitespace, and map labels.
    df["day_type"] = (
        df["day_type"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(DAY_TYPE_MAPPING)
    )

    # Reject missing or unsupported day-type categories.
    if df["day_type"].isna().any():
        raise ValueError("Unknown or missing day_type category found.")

    # Standardize line names by removing extra whitespace.
    df["line"] = df["line"].astype("string").str.strip()

    return df


def clean_training_data(input_path):
    """Clean original training data without modifying the source file."""

    # Load the raw dataset and record the original number of rows.
    df = pd.read_csv(input_path)
    original_rows = len(df)

    # Store statistics about the cleaning operations performed.
    report = {
        "original_rows": original_rows,
        "duplicate_rows_removed": 0,
        "sensor_values_corrected": 0,
        "categories_normalized": False,
    }

    # 1. Remove exact duplicate rows.
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    report["duplicate_rows_removed"] = duplicate_count

    # 2. Normalize category spelling.
    df = normalize_categories(df)
    report["categories_normalized"] = True

    # Convert the date column to datetime format for date-based filtering.
    df["date"] = pd.to_datetime(df["date"], errors="raise")

    # 3. Identify the known stuck sensor period.
    affected = (
        df["date"].between(STUCK_START, STUCK_END)
        & df["temp_c"].eq(STUCK_VALUE)
    )

    # Count the records identified as affected by the sensor issue.
    affected_count = int(affected.sum())

    # Use unaffected records as the reference dataset for temperature estimation.
    reference = df.loc[~affected].copy()

    # Calculate median temperature separately for each metro line.
    line_medians = reference.groupby("line")["temp_c"].median()

    # Ensure that a temperature reference exists for every line.
    if line_medians.isna().any():
        raise ValueError("Unable to calculate temperature median for every line.")

    # Correct the affected temperature readings using their line-specific median.
    df.loc[affected, "temp_c"] = (
        df.loc[affected, "line"].map(line_medians)
    )

    # Record the number of corrected readings and the replacement medians.
    report["sensor_values_corrected"] = affected_count
    report["temperature_medians_by_line"] = {
        str(line): float(value)
        for line, value in line_medians.items()
    }

    # 4. Final quality checks.
    if df.duplicated().any():
        raise ValueError("Duplicate rows remain after cleaning.")

    # Ensure the cleaned dataset contains no missing values.
    if df.isna().any().any():
        raise ValueError("Missing values found after cleaning.")

    # Verify that the day-type categories are standardized.
    if set(df["day_type"].unique()) != {
        "weekday", "weekend", "holiday"
    }:
        raise ValueError("Unexpected day_type categories.")

    # Validate the expected number of affected sensor readings.
    if affected_count != 117:
        raise ValueError(
            f"Expected 117 stuck readings after deduplication, "
            f"found {affected_count}."
        )

    # Add final dataset size and row-removal statistics to the report.
    report["final_rows"] = len(df)
    report["rows_removed"] = original_rows - len(df)

    return df, report


def clean_later_data(input_path):
    """Normalize categories in the clean later dataset."""

    # Load the later dataset and apply the same category normalization rules.
    df = pd.read_csv(input_path)
    df = normalize_categories(df)

    return df


def save_cleaning_outputs(df, report, output_dir="data/processed"):
    """Save processed data and a JSON cleaning report."""

    # Create the output directory if it does not exist.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the cleaned training dataset.
    df.to_csv(output_path / "train_clean.csv", index=False)

    # Save the cleaning statistics in JSON format.
    with open(
        output_path / "cleaning_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=4)

    print("Cleaning completed successfully.")
    print(json.dumps(report, indent=4))


if __name__ == "__main__":

    # Execute the cleaning pipeline using the original training dataset.
    cleaned_df, cleaning_report = clean_training_data(
        "data/raw/train.csv"
    )

    # Save the cleaned dataset and its cleaning report.
    save_cleaning_outputs(cleaned_df, cleaning_report)
```
