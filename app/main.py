import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from app.schemas import TransactionInput

app = FastAPI(title="Credit Card Fraud Detection API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
rf_path = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
log_path = os.path.join(BASE_DIR, "models", "logistic_model.pkl")
scaler_path = os.path.join(BASE_DIR, "data", "processed", "scaler.pkl")

# Load artifacts
# scaler = joblib.load(scaler_path)

# models = {
#     "rf": joblib.load(rf_path),
#     "logistic": joblib.load(log_path)
# }
scaler = joblib.load("data/processed/scaler.pkl")
rf_model = joblib.load("models/rf_model.pkl")


NUMERIC_TO_SCALE = ["Time", "Amount"]

PCA_FEATURES = [
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8",
    "V9", "V10", "V11", "V12", "V13", "V14", "V15",
    "V16", "V17", "V18", "V19", "V20", "V21", "V22",
    "V23", "V24", "V25", "V26", "V27", "V28"
]

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}

# @app.post("/predict")
# def predict_fraud(
#     data: TransactionInput,
#     model: str = Query("rf", enum=["rf", "logistic"])
# ):
#     selected_model = rf_model.get(model)

#     if selected_model is None:
#         raise HTTPException(status_code=400, detail="Invalid model selected")

#     input_data = np.array([list(data.dict().values())])
#     input_scaled = scaler.transform(input_data)

#     prediction = selected_model.predict(input_scaled)[0]

#     # Some models return predict_proba, some may not
#     if hasattr(selected_model, "predict_proba"):
#         probability = selected_model.predict_proba(input_scaled)[0][1]
#     else:
#         probability = None

#     return {
#         "model_used": model,
#         "fraud_prediction": int(prediction),
#         "fraud_probability": None if probability is None else round(float(probability), 4)
#     }
@app.post("/predict")
def predict_fraud(data: TransactionInput):
    input_data = np.array([list(data.dict().values())])
    input_scaled = scaler.transform(input_data)

    prediction = rf_model.predict(input_scaled)[0]
    probability = rf_model.predict_proba(input_scaled)[0][1]

    return {
        "model_used": "random_forest",
        "fraud_prediction": int(prediction),
        "fraud_probability": round(float(probability), 4)
    }


# python -m uvicorn app.main:app --reload

"""{
  "Time": 10000,
  "V1": -1.2,
  "V2": 0.5,
  "V3": -0.3,
  "V4": 1.1,
  "V5": -0.8,
  "V6": 0.2,
  "V7": -0.1,
  "V8": 0.05,
  "V9": -0.6,
  "V10": -1.4,
  "V11": 0.7,
  "V12": -0.9,
  "V13": 0.3,
  "V14": -2.1,
  "V15": 0.1,
  "V16": -0.4,
  "V17": -1.3,
  "V18": 0.2,
  "V19": -0.1,
  "V20": 0.01,
  "V21": 0.02,
  "V22": 0.04,
  "V23": -0.03,
  "V24": 0.5,
  "V25": -0.2,
  "V26": 0.1,
  "V27": 0.01,
  "V28": 0.02,
  "Amount": 123.45
}

"""