# Credit Card Fraud Detection API

A machine learning REST API that predicts whether a credit card transaction is fraudulent. Built with FastAPI and Scikit-learn, trained on 284,807 real transactions.

**Live demo:** https://credit-card-fraud-detection-87ku.onrender.com  
**API docs:** https://credit-card-fraud-detection-87ku.onrender.com/docs

> Note: Hosted on Render free tier — first request after inactivity may take 30–50 seconds to wake up.

---

## What it does

You send a credit card transaction as JSON, the API runs it through a trained ML model, and returns a fraud probability and risk level (LOW / MEDIUM / HIGH) in milliseconds.

In a real banking system, this API sits at the end of a transaction pipeline. The moment a card is swiped, the bank's internal systems collect signals, transform them into features, and call an API like this one to decide whether to approve or flag the transaction.

---

## Stack

- **Python 3.11**
- **FastAPI** — REST API framework
- **Scikit-learn** — Random Forest + Logistic Regression
- **Pandas / NumPy** — data handling
- **Joblib** — model serialization
- **Docker** — containerization
- **Render** — deployment

---

## Project structure

```
credit-card-fraud-detection/
├── app/
│   ├── main.py           # FastAPI app, endpoints
│   └── schemas.py        # Pydantic input schema
├── ml/
│   ├── train_rf.py       # Train Random Forest
│   ├── train_logistic.py # Train Logistic Regression
│   └── evaluate.py       # Evaluate both models
├── models/
│   ├── rf_model.pkl
│   └── logistic_model.pkl
├── data/
│   ├── raw/              # creditcard.csv (not in repo)
│   └── processed/        # scaler + test splits
├── static/
│   ├── index.html        # Landing page
│   └── style.css
├── Dockerfile
└── requirements.txt
```

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Landing page |
| `GET` | `/health` | Server and model status |
| `POST` | `/predict?model=rf` | Predict with Random Forest |
| `POST` | `/predict?model=logistic` | Predict with Logistic Regression |
| `GET` | `/explain?model=rf` | Top 10 most predictive features |
| `GET` | `/docs` | Interactive API explorer |

---

## Models

### Random Forest (`?model=rf`)
100 decision trees with `class_weight="balanced"`. Conservative and precise.

| Metric | Fraud class |
|--------|-------------|
| Precision | 96% |
| Recall | 74% |
| F1 Score | 84% |

When it flags fraud, it's right 96% of the time. Misses ~26% of fraud cases.

### Logistic Regression (`?model=logistic`)
Linear model with balanced class weights. Aggressive catch rate.

| Metric | Fraud class |
|--------|-------------|
| Precision | 6% |
| Recall | 90% |
| F1 Score | 12% |

Catches 90% of fraud but generates more false positives. Better for high-stakes environments where missing fraud is worse than false alarms.

---

## Dataset

[Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions from European cardholders
- 492 fraudulent (0.17%) — heavily imbalanced
- Features: `Time`, `Amount`, and `V1`–`V28` (PCA-transformed, anonymized)

The V1–V28 features are the result of PCA applied to the bank's raw transaction signals (location, device, merchant category, velocity, etc.) to protect customer privacy. No human types these values — they are generated automatically by the bank's internal systems.

---

## ML pipeline notes

- **No data leakage** — `StandardScaler` is fitted on training data only, then applied to test data using training statistics
- **Stratified split** — `train_test_split(..., stratify=y)` preserves the 0.17% fraud ratio in both splits
- **Class imbalance handled** — `class_weight="balanced"` on both models; evaluated on precision/recall for the fraud class, not accuracy
- **Evaluation metric** — accuracy is misleading on this dataset (a model that always predicts "not fraud" gets 99.83%). Fraud class recall is what matters.

---

## Running locally

```bash
# Clone the repo
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Install dependencies
pip install -r requirements.txt

# Download the dataset from Kaggle and place it at:
# data/raw/creditcard.csv

# Train the models
python ml/train_rf.py
python ml/train_logistic.py

# Start the API
uvicorn app.main:app --reload
```

Then open http://localhost:8000

---

## Running with Docker

```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```

---

## Example request

```bash
curl -X POST "https://credit-card-fraud-detection-87ku.onrender.com/predict?model=rf" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": -0.7976, "Amount": -0.3516,
    "V1": -0.6494, "V2": 1.4936, "V3": -1.8815,
    "V4": 1.6418, "V5": -0.9948, "V6": -0.7114,
    "V7": -2.4752, "V8": 0.9798, "V9": -2.0681,
    "V10": -4.4932, "V11": 2.2134, "V12": -4.7055,
    "V13": 0.6567, "V14": -6.4585, "V15": 0.6487,
    "V16": -5.5461, "V17": -7.743, "V18": -3.7183,
    "V19": 2.1069, "V20": 0.7284, "V21": 0.8879,
    "V22": 0.1125, "V23": -0.3529, "V24": -0.8645,
    "V25": 0.4314, "V26": 1.5691, "V27": 1.5629,
    "V28": 0.7648
  }'
```

Example response:

```json
{
  "model_used": "rf",
  "fraud_prediction": 1,
  "fraud_probability": 0.95,
  "risk_level": "HIGH",
  "decision": "Likely fraudulent transaction"
}
```

## Built by

Pranav Garg 
[github.com/pranavgarg13-star](https://github.com/pranavgarg13-star)