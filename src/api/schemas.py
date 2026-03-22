# Copyright (c) 2025 Deepika Sharma. Licensed under Apache 2.0.

from pydantic import BaseModel, Field
from typing import Optional

class TransactionRequest(BaseModel):
    """Input schema — one bank transaction to score for fraud."""
    Time: float = Field(..., description="Seconds elapsed since first transaction in dataset")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., description="Transaction amount in dollars")

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 0.0,
                "V1": -1.359807, "V2": -0.072781, "V3": 2.536347,
                "V4": 1.378155, "V5": -0.338321, "V6": 0.462388,
                "V7": 0.239599, "V8": 0.098698, "V9": 0.363787,
                "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
                "V13": -0.991390, "V14": -0.311169, "V15": 1.468177,
                "V16": -0.470401, "V17": 0.207971, "V18": 0.025791,
                "V19": 0.403993, "V20": 0.251412, "V21": -0.018307,
                "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
                "V25": 0.128539, "V26": -0.189115, "V27": 0.133558,
                "V28": -0.021053, "Amount": 149.62
            }
        }


class PredictionResponse(BaseModel):
    """Output schema — fraud score and verdict for a transaction."""
    is_fraud: bool = Field(..., description="True if transaction is predicted as fraud")
    fraud_probability: float = Field(..., description="Probability of fraud (0.0 to 1.0)")
    risk_level: str = Field(..., description="LOW, MEDIUM, or HIGH")
    amount: float = Field(..., description="Transaction amount from input")
    api_model_version: str = Field(default="xgboost-v1")


class HealthResponse(BaseModel):
    """Output schema for the /health endpoint."""
    status: str
    model_loaded: bool
    kafka_available: bool

class ExplainRequest(BaseModel):
    """Input schema for the /explain endpoint."""
    transaction_id: str = Field(..., description="Unique transaction reference")
    fraud_probability: float = Field(..., description="Score from /predict (0.0 to 1.0)")
    risk_level: str = Field(..., description="LOW, MEDIUM, or HIGH from /predict")
    amount: float = Field(..., description="Transaction amount")
    top_features: dict = Field(
        default_factory=dict,
        description="Key transaction features e.g. {'V14': -4.2, 'V12': -3.1}"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN-20240315-001",
                "fraud_probability": 0.9988,
                "risk_level": "HIGH",
                "amount": 149.62,
                "top_features": {"V14": -4.2, "V12": -3.1, "V4": 2.8, "V10": -2.5}
            }
        }


class ExplainResponse(BaseModel):
    """Output schema for the /explain endpoint — FCA-ready audit report."""
    transaction_id: str
    explanation: str = Field(..., description="Plain-English fraud analyst report")
    matched_pattern: str = Field(..., description="Closest fraud pattern from knowledge base")
    recommended_action: str = Field(..., description="Compliance team next step")
    confidence: str = Field(..., description="HIGH, MEDIUM, or LOW explanation confidence")
    prompt_version: str = Field(default="explain_v1")
    retrieved_context_preview: str = Field(..., description="First 120 chars of retrieved pattern")
