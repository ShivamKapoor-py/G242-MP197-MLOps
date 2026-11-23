from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


NUMERIC_FEATURES = [
    "hour",
    "nearby_offices",
    "temp_c",
    "rain_mm",
    "feeder_buses",
    "ticket_price_rs",
]

CATEGORICAL_FEATURES = [
    "day_type",
    "line",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor