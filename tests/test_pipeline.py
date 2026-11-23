```python
import pandas as pd
from pathlib import Path

# Import the model used to test the prediction pipeline.
from sklearn.ensemble import RandomForestRegressor

# Import Pipeline to combine preprocessing and model training into one workflow.
from sklearn.pipeline import Pipeline

# Import the data cleaning function used before training.
from src.mp197.data_cleaning import clean_training_data

# Import preprocessing tools and the expected input feature list.
from src.mp197.preprocessing import (
    build_preprocessor,
    FEATURE_COLUMNS,
)


# Test whether the training dataset is cleaned correctly.
def test_training_data_cleaning():
    # Run the cleaning pipeline on the original training dataset.
    df, report = clean_training_data("data/raw/train.csv")

    # Verify that the dataset contains the expected number of rows after cleaning.
    assert len(df) == 1000

    # Check that the expected number of duplicate rows were removed.
    assert report["duplicate_rows_removed"] == 55

    # Check that the known faulty temperature readings were corrected.
    assert report["sensor_values_corrected"] == 117

    # Ensure that no duplicate records remain in the cleaned dataset.
    assert not df.duplicated().any()

    # Ensure that the cleaned dataset contains no missing values.
    assert df.isnull().sum().sum() == 0


# Test whether categorical values are standardized correctly.
def test_categories_are_normalized():
    # Clean the training dataset to obtain normalized categories.
    df, _ = clean_training_data("data/raw/train.csv")

    # Verify that day_type contains only the expected category labels.
    assert set(df["day_type"].unique()) == {
        "weekday",
        "weekend",
        "holiday",
    }


# Test whether the preprocessing function transforms input features correctly.
def test_preprocessor_output():
    # Create a small sample dataset with numerical and categorical features.
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

    # Create the preprocessing transformer.
    preprocessor = build_preprocessor()

    # Fit the transformer on the sample data and convert features.
    transformed = preprocessor.fit_transform(sample)

    # Check that both sample rows are preserved after transformation.
    assert transformed.shape[0] == 2

    # One-hot encoding should create additional columns for categories.
    assert transformed.shape[1] > len(FEATURE_COLUMNS)


# Test whether the complete preprocessing and model pipeline can make predictions.
def test_model_prediction_pipeline():
    # Load the cleaned training data.
    df, _ = clean_training_data("data/raw/train.csv")

    # Select the input features for the first 50 records.
    X = df[FEATURE_COLUMNS].head(50)

    # Select the corresponding target values.
    y = df["entries_per_hour"].head(50)

    # Combine preprocessing and Random Forest model into a single pipeline.
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=10,
                random_state=42,
            ),
        ),
    ])

    # Train the pipeline using the selected features and target values.
    pipeline.fit(X, y)

    # Generate predictions using the trained pipeline.
    predictions = pipeline.predict(X)

    # Verify that one prediction is generated for each input record.
    assert len(predictions) == 50

    # Ensure that all generated predictions are valid and not missing.
    assert pd.Series(predictions).notna().all()
```
