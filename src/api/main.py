from fastapi import FastAPI
import pandas as pd
import mlflow.pyfunc

from src.api.pydantic_models import (
    PredictionRequest,
    PredictionResponse
)

app = FastAPI(
    title="Credit Risk Prediction API",
    version="1.0"
)

# Load model from MLflow Model Registry
MODEL_NAME = "credit_risk_model"
MODEL_VERSION = "1"

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}"
)


@app.get("/")
def home():
    return {"message": "Credit Risk API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    data = pd.DataFrame([{
        "TotalTransactionAmount": request.TotalTransactionAmount,
        "AverageTransactionAmount": request.AverageTransactionAmount,
        "TransactionCount": request.TransactionCount,
        "StdTransactionAmount": request.StdTransactionAmount,
        "Recency": request.Recency,
        "Frequency": request.Frequency,
        "Monetary": request.Monetary
    }])

    prediction = model.predict(data)

    probability = float(prediction[0])

    return PredictionResponse(
        risk_probability=probability,
        prediction=int(probability >= 0.5)
    )