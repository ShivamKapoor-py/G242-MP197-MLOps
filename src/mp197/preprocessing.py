# Import tools for applying transformations to selected columns.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# Numerical features are passed to the model as numeric values.
NUMERIC_FEATURES = [
    "hour",
    "nearby_offices",
    "temp_c",
    "rain_mm",
    "feeder_buses",
    "ticket_price_rs",
]

# Categorical features contain labels that need numerical encoding.
CATEGORICAL_FEATURES = [
    "day_type",
    "line",
]

# Combine both feature groups into the complete model input feature list.
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor():
    # ColumnTransformer allows different preprocessing for different columns.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                # Convert categorical values into numerical indicator columns.
                "categorical",
                OneHotEncoder(
                    # Ignore categories that were not seen during training.
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                # Keep numerical features in their original numerical form.
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
        ],
        # Drop columns that are not explicitly included in the transformers.
        remainder="drop",
    )

    # Return the preprocessing pipeline for use during training and prediction.
    return preprocessor
