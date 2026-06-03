from pydantic import BaseModel


class PredictionRequest(BaseModel):
    TotalTransactionAmount: float
    AverageTransactionAmount: float
    TransactionCount: int
    StdTransactionAmount: float
    Recency: int
    Frequency: int
    Monetary: float


class PredictionResponse(BaseModel):
    risk_probability: float
    prediction: int