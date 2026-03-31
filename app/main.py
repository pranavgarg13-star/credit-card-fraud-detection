from fastapi import FastAPI
from fastapi import Query, HTTPException
import joblib
import numpy as np
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
from app.schemas import TransactionInput
app = FastAPI()

# After app = FastAPI(...), add:
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCALER_PATH = os.path.join(BASE_DIR, "data", "processed", "scaler.pkl")


scaler = joblib.load(SCALER_PATH)

RF_MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.pkl")
LOG_MODEL_PATH = os.path.join(BASE_DIR, "models", "logistic_model.pkl")
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



@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": list(MODEL_REGISTRY.keys())
    }

@app.post("/predict")
def predict_fraud(
    data: TransactionInput,
    model: str = Query("rf", enum=["rf", "logistic"])
):
    selected_model = MODEL_REGISTRY.get(model)

    if not selected_model:
        raise HTTPException(status_code=400, detail="Invalid model selected")

    input_dict = data.dict()
    
    X = pd.DataFrame([input_dict])[FEATURE_ORDER]


    try:
        X_scaled = scaler.transform(X)
        prediction = selected_model.predict(X_scaled)[0]
        probability = selected_model.predict_proba(X_scaled)[0][1]
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
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

@app.get("/explain")
def explain_model(model: str = Query("rf", enum=["rf", "logistic"])):
    selected_model = MODEL_REGISTRY.get(model)
    if not selected_model:
        raise HTTPException(status_code=400, detail="Invalid model")
    
    if model == "rf":
        importances = selected_model.feature_importances_
        features = FEATURE_ORDER
        ranked = sorted(
            zip(features, importances),
            key=lambda x: x[1],
            reverse=True
        )
        return {
            "model": model,
            "top_features": [
                {"feature": f, "importance": round(float(i), 4)}
                for f, i in ranked[:10]
            ]
        }
    else:
        return {"model": model, "note": "Logistic regression uses coefficients, not importances"}




