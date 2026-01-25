import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os





# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "creditcard.csv")
SCALER_PATH = os.path.join(BASE_DIR, "data", "processed", "scaler.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.pkl")

# Load data
df = pd.read_csv(DATA_PATH)

# Features & target
X = df.drop("Class", axis=1)   # 30 features
y = df["Class"]

# Scale ALL features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, SCALER_PATH)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Train RF
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Save model
joblib.dump(rf, MODEL_PATH)

print("✅ Random Forest trained and saved")
print("✅ Scaler saved")
