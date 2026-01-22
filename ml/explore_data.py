import pandas as pd
import os

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Build absolute path to dataset
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "creditcard.csv")

print("Looking for dataset at:")
print(DATA_PATH)

df = pd.read_csv(DATA_PATH)

print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
print("\nClass distribution:")
print(df["Class"].value_counts())
