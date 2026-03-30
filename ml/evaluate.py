import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

X_test = joblib.load(os.path.join(BASE_DIR, "data/processed/X_test.pkl"))
y_test = joblib.load(os.path.join(BASE_DIR, "data/processed/y_test.pkl"))

models = {
    "Random Forest":  os.path.join(BASE_DIR, "models/rf_model.pkl"),
    "Logistic Regression": os.path.join(BASE_DIR, "models/logistic_model.pkl"),
}

for name, path in models.items():
    model = joblib.load(path)
    y_pred = model.predict(X_test)
    print(f"\n{'='*40}")
    print(f"{name}")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))