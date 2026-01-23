import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load processed data
X_train = joblib.load(os.path.join(BASE_DIR, "data/processed/X_train.pkl"))
X_test = joblib.load(os.path.join(BASE_DIR, "data/processed/X_test.pkl"))
y_train = joblib.load(os.path.join(BASE_DIR, "data/processed/y_train.pkl"))
y_test = joblib.load(os.path.join(BASE_DIR, "data/processed/y_test.pkl"))

# Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Train
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
print("\n📌 Random Forest Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
joblib.dump(rf_model, os.path.join(BASE_DIR, "models/random_forest_model.pkl"))

print("✅ Random Forest model trained and saved!")
