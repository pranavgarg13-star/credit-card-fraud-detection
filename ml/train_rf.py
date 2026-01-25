import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data","raw" ,"creditcard.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "data", "processed", "scaler.pkl")

# Ensure folders exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)

# ---------------- Load Data ----------------
df = pd.read_csv(DATA_PATH)

X = df.drop("Class", axis=1)
y = df["Class"]

# ---------------- Scale ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- Train ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ---------------- Save ----------------
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("✅ Random Forest model saved to:", MODEL_PATH)
print("✅ Scaler saved to:", SCALER_PATH)
