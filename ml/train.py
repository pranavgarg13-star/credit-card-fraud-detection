import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

# Load processed data
X_train = pd.read_csv("../data/processed/X_train.csv")
y_train = pd.read_csv("../data/processed/y_train.csv").values.ravel()

# Initialize model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1
)

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "../models/logistic_model.pkl")

print("Model trained and saved successfully!")
