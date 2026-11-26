from pathlib import Path
import pytest
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.mp197.data_cleaning import clean_training_data
from src.mp197.preprocessing import build_preprocessor, FEATURE_COLUMNS


@pytest.mark.skipif(
    not Path("data/raw/train.csv").exists(),
    reason="Full training dataset is unavailable in this CI environment",
)
def test_training_data_cleaning():
    df, report = clean_training_data("data/raw/train.csv")

    assert len(df) == 1000
    assert report["duplicate_rows_removed"] == 55
    assert report["sensor_values_corrected"] == 117
    assert not df.duplicated().any()
    assert df.isna().sum().sum() == 0


@pytest.mark.skipif(
    not Path("data/raw/train.csv").exists(),
    reason="Full training dataset is unavailable in this CI environment",
)
def test_categories_are_normalized():
    df, _ = clean_training_data("data/raw/train.csv")

    assert set(df["day_type"].unique()) == {
        "weekday", "weekend", "holiday"
    }


def test_preprocessor_output():
    sample = pd.DataFrame({
        "hour": [8, 9],
        "nearby_offices": [5, 7],
        "temp_c": [26.5, 28.0],
        "rain_mm": [0.0, 2.0],
        "feeder_buses": [3, 4],
        "ticket_price_rs": [25, 30],
        "day_type": ["weekday", "weekend"],
        "line": ["Blue Line", "Red Line"],
    })

    transformed = build_preprocessor().fit_transform(sample)

    assert transformed.shape[0] == 2
    assert transformed.shape[1] > len(FEATURE_COLUMNS)


@pytest.mark.skipif(
    not Path("data/raw/train.csv").exists(),
    reason="Full training dataset is unavailable in this CI environment",
)
def test_model_prediction_pipeline():
    df, _ = clean_training_data("data/raw/train.csv")

    X = df[FEATURE_COLUMNS].head(50)
    y = df["entries_per_hour"].head(50)

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", RandomForestRegressor(
            n_estimators=10,
            random_state=42,
        )),
    ])

    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    assert len(predictions) == 50
    assert pd.Series(predictions).notna().all()
    assert (predictions >= 0).all()
