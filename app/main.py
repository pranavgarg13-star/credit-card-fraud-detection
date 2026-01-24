import os
import joblib
import numpy as np
from fastapi import FastAPI
from app.schemas import TransactionInput

app = FastAPI(title="Credit Card Fraud Detection API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "models", "random_forest_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "data", "processed", "scaler.pkl"))

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

@app.post("/predict")
def predict_fraud(data: TransactionInput):

    # 1️⃣ Extract Time & Amount → scale
    to_scale = np.array([[data.Time, data.Amount]])
    scaled_values = scaler.transform(to_scale)

    # 2️⃣ Extract PCA features (NO scaling)
    pca_values = np.array([[getattr(data, f) for f in PCA_FEATURES]])

    # 3️⃣ Combine in training order
    final_input = np.hstack([scaled_values, pca_values])

    # 4️⃣ Predict
    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0][1]

    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": round(float(probability), 4)
    }
