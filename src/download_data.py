
import pandas as pd
from pathlib import Path
from ucimlrepo import fetch_ucirepo

# Locate the main project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Create dataset directory
DATA_FOLDER = PROJECT_ROOT / "data" / "raw"
DATA_FOLDER.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("PHISHING WEBSITE DETECTION PROJECT")
print("Downloading PhiUSIIL Dataset")
print("=" * 55)

# Download dataset from UCI
print("\nConnecting to UCI repository...")

dataset = fetch_ucirepo(id=967)

# Extract features and target
X = dataset.data.features
y = dataset.data.targets

# Combine features and target
df = pd.concat([X, y], axis=1)

# Save the original dataset
output_file = DATA_FOLDER / "phiusiil.csv"

df.to_csv(output_file, index=False)

print("\nDataset downloaded successfully!")

# Dataset information
print("\nDATASET INFORMATION")
print("-" * 40)

print("Total records:", df.shape[0])
print("Total columns:", df.shape[1])

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nFIRST FIVE RECORDS:")
print(df.head())

print("\nMISSING VALUES:")
print(df.isnull().sum().sum())

print("\nTARGET DISTRIBUTION:")
print(y.value_counts())

print("\nDataset saved at:")
print(output_file)

print("\nDOWNLOAD COMPLETED!")
