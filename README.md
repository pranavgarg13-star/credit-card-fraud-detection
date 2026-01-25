# Credit Card Fraud Detection API

This project implements a machine learning-based credit card fraud detection system
and deploys it using FastAPI and Docker.

## Problem Statement
Credit card fraud is a major issue in financial systems. The goal of this project
is to predict whether a transaction is fraudulent based on transaction features.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Random Forest Classifier
- FastAPI
- Docker


## Project Structure
  credit-card-fraud-detection/
    │
    ├── app/
│        └── main.py            # FastAPI application
│
├── ml/
│   └── train_rf.py        # Model training script
│
├── models/
│   └── rf_model.pkl       # Trained model
│
├── data/
│   ├──raw/
│   │   └──creditcard.csv
│   └── processed/
│       └── scaler.pkl
│
├── requirements.txt
├── README.md
└── .gitignore

## How to Run the Project

### 1. Install dependencies
pip install -r requirements.txt

### 2. Train the model
python ml/train_rf.py

### 3. Run the API
python -m uvicorn app.main:app --reload

### 4. Test the API
Open:
http://127.0.0.1:8000/docs
