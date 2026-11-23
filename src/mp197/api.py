
import json
import logging
import sys

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


# --------------------------------------------------
# Structured JSON Logging
# --------------------------------------------------

class JsonFormatter(logging.Formatter):

    def format(self, record):
        payload = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        # Add structured event fields directly to JSON
        if hasattr(record, "event_data"):
            payload.update(record.event_data)

        return json.dumps(payload)


logger = logging.getLogger("mp197_api")
logger.setLevel(logging.INFO)
logger.propagate = False

# Avoid adding duplicate handlers during repeated imports
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)


# --------------------------------------------------
# MLflow Model Loading
# --------------------------------------------------

model = mlflow.sklearn.load_model("models/champion")


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="MP197 Metro Crowd Prediction API",
    version="1.0.0",
    description="Predicts hourly metro station entries."
)


# --------------------------------------------------
# Request Schema
# --------------------------------------------------

class PredictionRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    hour: int = Field(ge=0, le=23)
    nearby_offices: int = Field(ge=0)
    temp_c: float
    rain_mm: float = Field(ge=0)
    feeder_buses: int = Field(ge=0)
    ticket_price_rs: int = Field(ge=0, le=95)

    day_type: Literal["weekday", "weekend", "holiday"]

    line: Literal["Blue Line", "Red Line", "Yellow Line"]


# --------------------------------------------------
# Response Schema
# --------------------------------------------------

class PredictionResponse(BaseModel):
    predicted_entries_per_hour: float


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    input_data = request.model_dump()

    input_df = pd.DataFrame([input_data])

    prediction = float(model.predict(input_df)[0])

    # Log a structured JSON prediction event
    logger.info(
        "prediction",
        extra={
            "event_data": {
                "event": "prediction",
                "line": request.line,
                "hour": request.hour,
                "predicted_entries_per_hour": prediction,
            }
        }
    )

    return {
        "predicted_entries_per_hour": prediction
    }
