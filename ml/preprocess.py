import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR,"data","raw","creditcard.csv")

df = pd.read_csv(DATA_PATH)

X = df.drop("Class",axis=1)
y = df["Class"]

X_train , X_test , y_train , y_test = train_test_split(
    X , y ,
    test_size= 0.2 ,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train[["Time","Amount"]] = scaler.fit_transform(
    X_train[["Time","Amount"]]
)

X_test[["Time","Amount"]] = scaler.transform(
    X_test[["Time","Amount"]]
)

PROCESSED_DIR = os.path.join(BASE_DIR,"data","processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

joblib.dump(X_train, os.path.join(PROCESSED_DIR, "X_train.pkl"))
joblib.dump(X_test, os.path.join(PROCESSED_DIR, "X_test.pkl"))
joblib.dump(y_train, os.path.join(PROCESSED_DIR, "y_train.pkl"))
joblib.dump(y_test, os.path.join(PROCESSED_DIR, "y_test.pkl"))
joblib.dump(scaler, os.path.join(PROCESSED_DIR, "scaler.pkl"))

print("Preprocessing completed successfully!")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
