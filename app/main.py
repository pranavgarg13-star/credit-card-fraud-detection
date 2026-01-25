from fastapi import FastAPI
from fastapi import Query, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os


app = FastAPI()

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "data", "processed", "scaler.pkl")

# Load model & scaler ONCE
# rf_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

RF_MODEL_PATH = "models/rf_model.pkl"
LOG_MODEL_PATH = "models/logistic_model.pkl"

rf_model = joblib.load(RF_MODEL_PATH)
logistic_model = joblib.load(LOG_MODEL_PATH)

MODEL_REGISTRY = {
    "rf": rf_model,
    "logistic": logistic_model
}


FEATURE_ORDER = [
    "Time",
    "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26","V27","V28",
    "Amount"
]

HIGH_RISK_THRESHOLD = 0.8
MEDIUM_RISK_THRESHOLD = 0.5

# Input schema
from pydantic import BaseModel

class Transaction(BaseModel):
    Time: float
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
    Amount: float


@app.get("/")
def home():
    return {"message": "Credit Card Fraud Detection API is running"}

@app.post("/predict")
def predict_fraud(
    data: Transaction,
    model: str = Query("rf", enum=["rf", "logistic"])
):
    selected_model = MODEL_REGISTRY.get(model)

    if not selected_model:
        raise HTTPException(status_code=400, detail="Invalid model selected")

    input_dict = data.dict()
    X = np.array(
        [input_dict[feature] for feature in FEATURE_ORDER]
    ).reshape(1, -1)

    X_scaled = scaler.transform(X)

    prediction = selected_model.predict(X_scaled)[0]
    probability = selected_model.predict_proba(X_scaled)[0][1]
    prob = float(probability)

    if prob >= HIGH_RISK_THRESHOLD:
       risk = "HIGH"
       decision = "Likely fraudulent transaction"
    elif prob >= MEDIUM_RISK_THRESHOLD:
       risk = "MEDIUM"
       decision = "Suspicious transaction – review recommended"
    else:
        risk = "LOW"
        decision = "Likely legitimate transaction"
    return {
         "model_used": model,
    "fraud_prediction": int(prediction),
    "fraud_probability": round(prob, 4),
    "risk_level": risk,
    "decision": decision
    }




