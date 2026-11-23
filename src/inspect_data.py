
import pandas as pd
from pathlib import Path

# Locate project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load original dataset
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "phiusiil.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("PHIUSIIL DATASET EXPLORATION")
print("=" * 60)

# 1. Dataset dimensions
print("\n1. DATASET SHAPE")
print(df.shape)

# 2. Column names
print("\n2. ALL COLUMN NAMES")
for column in df.columns:
    print(column)

# 3. First five records
print("\n3. FIRST FIVE RECORDS")
print(df.head().to_string())

# 4. Data types
print("\n4. DATA TYPES")
print(df.dtypes)

# 5. Missing values
print("\n5. MISSING VALUES")
print(df.isnull().sum().sum())

# 6. Duplicate records
print("\n6. DUPLICATE RECORDS")
print(df.duplicated().sum())

# 7. Target distribution
print("\n7. TARGET DISTRIBUTION")

if "label" in df.columns:
    print(df["label"].value_counts())
else:
    print("Target column not identified yet.")

print("\nEXPLORATION COMPLETED!")
