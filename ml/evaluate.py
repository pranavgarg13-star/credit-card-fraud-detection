import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

X_test = joblib.load(os.path.join(BASE_DIR, "data/processed/X_test.pkl"))
y_test = joblib.load(os.path.join(BASE_DIR, "data/processed/y_test.pkl"))

model = joblib.load(os.path.join(BASE_DIR, "models/logistic_model.pkl"))

y_pred = model.predict(X_test)

print("\n📌 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n📌 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
