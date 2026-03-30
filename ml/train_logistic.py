import os
import joblib
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load processed data
X_train = joblib.load(os.path.join(BASE_DIR, "data/processed/X_train.pkl"))
y_train = joblib.load(os.path.join(BASE_DIR, "data/processed/y_train.pkl"))

# Train model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

# Save model
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
joblib.dump(model, os.path.join(BASE_DIR, "models/logistic_model.pkl"))

print("✅ Model trained and saved successfully!")
