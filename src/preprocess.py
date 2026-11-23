
import pandas as pd
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "phiusiil.csv"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "processed"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_FOLDER / "url_dataset.csv"

print("=" * 55)
print("PHISHING DATA PREPROCESSING")
print("=" * 55)

# Load original dataset
df = pd.read_csv(INPUT_FILE)

print("\nOriginal dataset shape:", df.shape)

# Check required columns
required_columns = ["URL", "label"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

# Select only URL and label
df = df[["URL", "label"]].copy()

# Remove missing values
df = df.dropna(subset=["URL", "label"])

# Clean URL whitespace
df["URL"] = df["URL"].astype(str).str.strip()

# Remove empty URLs
df = df[df["URL"] != ""]

# Convert labels to integers
df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

# Verify label values
if not set(df["label"].unique()).issubset({0, 1}):
    raise ValueError("Unexpected label values found!")

# Check for conflicting labels on the same URL
conflicts = df.groupby("URL")["label"].nunique()
conflicts = conflicts[conflicts > 1]

if not conflicts.empty:
    raise ValueError(
        f"Found {len(conflicts)} URLs with conflicting labels. "
        "Inspect them before continuing."
    )

# Remove repeated URLs with the same label
before = len(df)
df = df.drop_duplicates(subset=["URL"])
removed = before - len(df)

# Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\nPREPROCESSING RESULTS")
print("-" * 40)

print("Remaining records:", len(df))
print("Repeated URLs removed:", removed)

print("\nLabel distribution:")
print(df["label"].value_counts().sort_index())

print("\nFirst five cleaned records:")
print(df.head())

print("\nSaved to:", OUTPUT_FILE)

print("\nPREPROCESSING COMPLETED!")
