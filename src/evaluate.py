
import pandas as pd
import numpy as np
import joblib
import json
import ipaddress
import tldextract
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from urllib.parse import urlparse

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

FEATURE_FILE = ROOT / "data" / "processed" / "url_features.csv"

EMBEDDING_FILE = (
    ROOT / "data" / "processed" / "embeddings" / "url_embeddings.npy"
)

METADATA_FILE = (
    ROOT / "data" / "processed" / "embeddings" / "embedding_metadata.csv"
)

MODEL_FOLDER = ROOT / "models"
REPORT_FOLDER = ROOT / "reports"
FIGURE_FOLDER = REPORT_FOLDER / "figures"

FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHISHING DETECTION - MODEL EVALUATION")
print("=" * 60)


# ==========================================
# 2. LOAD DATA
# ==========================================

df = pd.read_csv(FEATURE_FILE)
metadata = pd.read_csv(METADATA_FILE)

embeddings = np.load(EMBEDDING_FILE, mmap_mode="r")

print("\nFeature dataset:", df.shape)
print("Embedding shape:", embeddings.shape)

# Verify alignment
if not (
    len(df) == len(metadata) == embeddings.shape[0]
):
    raise ValueError("Dataset alignment error!")

if not df["URL"].reset_index(drop=True).equals(
    metadata["URL"].reset_index(drop=True)
):
    raise ValueError("URL alignment mismatch!")

if not df["label"].reset_index(drop=True).equals(
    metadata["label"].reset_index(drop=True)
):
    raise ValueError("Label alignment mismatch!")

print("\nDataset alignment verified!")


# ==========================================
# 3. PREPARE FEATURES
# ==========================================

y = df["label"].astype(int).to_numpy()

feature_columns = [
    col for col in df.columns
    if col not in ["URL", "label"]
]

X_handcrafted = df[feature_columns].fillna(0).to_numpy(
    dtype=np.float32
)


# ==========================================
# 4. RECREATE DOMAIN GROUPS
# ==========================================

extractor = tldextract.TLDExtract(suffix_list_urls=None)

def get_domain(url):

    try:
        parsed = urlparse(
            url if "://" in url else "//" + url
        )

        hostname = parsed.hostname or ""

        try:
            ipaddress.ip_address(hostname)
            return hostname
        except ValueError:
            pass

        result = extractor(hostname)

        if result.top_domain_under_public_suffix:
            return result.top_domain_under_public_suffix

        return hostname

    except Exception:
        return "unknown"


groups = np.array([
    get_domain(url) for url in df["URL"].astype(str)
])

if np.any(groups == "unknown"):
    raise ValueError("Unknown domains found!")


# ==========================================
# 5. RECREATE ORIGINAL TRAIN TEST SPLIT
# ==========================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X_handcrafted, y, groups)
)

overlap = set(groups[train_idx]) & set(groups[test_idx])

if overlap:
    raise ValueError("Domain overlap detected!")

print("\nTraining records:", len(train_idx))
print("Testing records:", len(test_idx))
print("Overlapping domains:", len(overlap))


# ==========================================
# 6. LOAD SAVED MODEL AND SCALER
# ==========================================

print("\nLoading saved models...")

model = joblib.load(
    MODEL_FOLDER / "hybrid_random_forest.pkl"
)

scaler = joblib.load(
    MODEL_FOLDER / "feature_scaler.pkl"
)

print("Hybrid Random Forest loaded!")
print("Feature scaler loaded!")


# ==========================================
# 7. PREPARE TEST FEATURES
# ==========================================

X_test_emb = np.asarray(
    embeddings[test_idx],
    dtype=np.float32
)

X_test_hand = scaler.transform(
    X_handcrafted[test_idx]
).astype(np.float32)

X_test_hybrid = np.concatenate(
    [X_test_emb, X_test_hand],
    axis=1
)

y_test = y[test_idx]

print("\nTest feature shape:", X_test_hybrid.shape)


# ==========================================
# 8. GENERATE PREDICTIONS
# ==========================================

predictions = model.predict(X_test_hybrid)

# Model class 0 = Phishing
# Model class 1 = Legitimate

phishing_index = list(model.classes_).index(0)

phishing_probability = model.predict_proba(
    X_test_hybrid
)[:, phishing_index]

# Convert predictions into phishing-positive format
y_phishing = (y_test == 0).astype(int)

predicted_phishing = (predictions == 0).astype(int)


# ==========================================
# 9. CALCULATE METRICS
# ==========================================

metrics = {
    "Accuracy": accuracy_score(y_test, predictions),

    "Phishing Precision": precision_score(
        y_phishing,
        predicted_phishing,
        zero_division=0
    ),

    "Phishing Recall": recall_score(
        y_phishing,
        predicted_phishing,
        zero_division=0
    ),

    "Phishing F1 Score": f1_score(
        y_phishing,
        predicted_phishing,
        zero_division=0
    ),

    "Phishing ROC AUC": roc_auc_score(
        y_phishing,
        phishing_probability
    ),

    "Phishing Average Precision": average_precision_score(
        y_phishing,
        phishing_probability
    )
}

print("\n" + "=" * 60)
print("FINAL EVALUATION RESULTS")
print("=" * 60)

for name, value in metrics.items():
    print(f"{name}: {value:.5f}")

# Save metrics
with open(
    REPORT_FOLDER / "evaluation_metrics.json",
    "w"
) as file:

    json.dump(metrics, file, indent=4)


# ==========================================
# 10. CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Phishing", "Legitimate"],
    zero_division=0
)

print("\nCLASSIFICATION REPORT")
print(report)

with open(
    REPORT_FOLDER / "classification_report.txt",
    "w"
) as file:

    file.write(report)


# ==========================================
# 11. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Phishing", "Legitimate"],
    yticklabels=["Phishing", "Legitimate"]
)

plt.title("Hybrid Random Forest - Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.savefig(
    FIGURE_FOLDER / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 12. ROC CURVE
# ==========================================

fpr, tpr, thresholds = roc_curve(
    y_phishing,
    phishing_probability
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Hybrid Random Forest (AUC = {metrics['Phishing ROC AUC']:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve - Phishing Detection")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.savefig(
    FIGURE_FOLDER / "roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 13. PRECISION RECALL CURVE
# ==========================================

precision, recall, thresholds = precision_recall_curve(
    y_phishing,
    phishing_probability
)

plt.figure(figsize=(8, 6))

plt.plot(
    recall,
    precision,
    label=f"Average Precision = {metrics['Phishing Average Precision']:.4f}"
)

plt.title("Precision-Recall Curve - Phishing Detection")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()

plt.savefig(
    FIGURE_FOLDER / "precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 14. FEATURE IMPORTANCE
# ==========================================

feature_names = (
    [f"embedding_{i+1}" for i in range(X_test_emb.shape[1])]
    + feature_columns
)

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_df.to_csv(
    REPORT_FOLDER / "feature_importance.csv",
    index=False
)

top_features = importance_df.head(20)

plt.figure(figsize=(10, 8))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title("Top 20 Important Features")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.savefig(
    FIGURE_FOLDER / "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 15. SAVE TEST PREDICTIONS
# ==========================================

prediction_df = pd.DataFrame({
    "URL": df["URL"].iloc[test_idx].to_numpy(),
    "Actual_Label": y_test,
    "Predicted_Label": predictions,
    "Phishing_Probability": phishing_probability
})

prediction_df.to_csv(
    REPORT_FOLDER / "test_predictions.csv",
    index=False
)


# ==========================================
# 16. COMPLETION
# ==========================================

print("\n" + "=" * 60)
print("MODEL EVALUATION COMPLETED!")
print("=" * 60)

print("\nGraphs saved in:", FIGURE_FOLDER)

print("\nReports saved in:", REPORT_FOLDER)

print("\nAll evaluation files generated successfully!")
