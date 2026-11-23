
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "url_dataset.csv"

FIGURE_FOLDER = PROJECT_ROOT / "reports" / "figures"
FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)

# Load processed dataset
df = pd.read_csv(DATA_PATH)

# Set graph style
sns.set_theme(style="whitegrid")

print("=" * 50)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 50)

print("\nDataset shape:", df.shape)

# 1. Target distribution
print("\n1. TARGET DISTRIBUTION")
print(df["label"].value_counts())

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="label")

plt.title("Phishing vs Legitimate URL Distribution")
plt.xlabel("Class (0 = Phishing, 1 = Legitimate)")
plt.ylabel("Number of URLs")

plt.savefig(FIGURE_FOLDER / "class_distribution.png",
            dpi=300, bbox_inches="tight")
plt.show()

# 2. URL length analysis
df["url_length"] = df["URL"].astype(str).apply(len)

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="url_length",
    hue="label",
    bins=50,
    element="step",
    stat="density",
    common_norm=False
)

plt.title("URL Length Distribution")
plt.xlabel("URL Length")
plt.ylabel("Density")

plt.savefig(FIGURE_FOLDER / "url_length_distribution.png",
            dpi=300, bbox_inches="tight")
plt.show()

# 3. Boxplot of URL lengths
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="label",
    y="url_length"
)

plt.title("URL Length Comparison")
plt.xlabel("Class (0 = Phishing, 1 = Legitimate)")
plt.ylabel("URL Length")

plt.savefig(FIGURE_FOLDER / "url_length_boxplot.png",
            dpi=300, bbox_inches="tight")
plt.show()

# 4. Basic URL statistics
print("\n2. URL LENGTH STATISTICS")

print(
    df.groupby("label")["url_length"]
    .agg(["mean", "median", "min", "max"])
)

# 5. Save analysis dataset
df.to_csv(
    PROJECT_ROOT / "data" / "processed" / "url_dataset_with_length.csv",
    index=False
)

print("\nEDA COMPLETED SUCCESSFULLY!")
print("Graphs saved in:", FIGURE_FOLDER)
