
import json
import joblib
import numpy as np
import pandas as pd
import tldextract

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
    PROJECT_ROOT / "data" / "processed" / "url_features.csv"
)

METADATA_FILE = (
    PROJECT_ROOT / "data" / "processed" / "embeddings"
    / "embedding_metadata.csv"
)

EMBEDDING_FILE = (
    PROJECT_ROOT / "data" / "processed" / "embeddings"
    / "url_embeddings.npy"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. CONFIGURATION
# ==========================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

N_ESTIMATORS = 100
MAX_DEPTH = 20
MIN_SAMPLES_LEAF = 2

# Labels:
# 0 = Phishing
# 1 = Legitimate


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def find_column(dataframe, expected_name):

    for column in dataframe.columns:

        if str(column).strip().lower() == expected_name.lower():
            return column

    raise ValueError(
        f"Column '{expected_name}' not found. "
        f"Available columns: {list(dataframe.columns)}"
    )


def normalize_url(value):

    if pd.isna(value):
        return ""

    return str(value).strip()


def extract_domain(url):

    try:

        extracted = tldextract.extract(url)

        domain = extracted.top_domain_under_public_suffix

        if domain:
            return domain.lower()

        hostname = extracted.fqdn

        return hostname.lower() if hostname else url.lower()

    except Exception:

        return url.lower()


# ==========================================
# 4. LOAD DATASETS
# ==========================================

print("\nLoading PhishGuard training data...\n")

if not FEATURE_FILE.exists():
    raise FileNotFoundError(FEATURE_FILE)

if not METADATA_FILE.exists():
    raise FileNotFoundError(METADATA_FILE)

if not EMBEDDING_FILE.exists():
    raise FileNotFoundError(EMBEDDING_FILE)


df = pd.read_csv(FEATURE_FILE)

metadata = pd.read_csv(METADATA_FILE)

embeddings = np.load(
    EMBEDDING_FILE,
    mmap_mode="r"
)

print("Feature dataset:", df.shape)
print("Metadata dataset:", metadata.shape)
print("Embedding dataset:", embeddings.shape)


# ==========================================
# 5. VALIDATE DATASET COLUMNS
# ==========================================

url_column = find_column(df, "url")
label_column = find_column(df, "label")

metadata_url_column = find_column(metadata, "url")
metadata_label_column = find_column(metadata, "label")


# ==========================================
# 6. VERIFY DATA ALIGNMENT
# ==========================================

print("\nChecking dataset alignment...")

feature_urls = df[url_column].map(normalize_url).to_numpy()

metadata_urls = (
    metadata[metadata_url_column]
    .map(normalize_url)
    .to_numpy()
)

feature_labels = (
    df[label_column]
    .astype(int)
    .to_numpy()
)

metadata_labels = (
    metadata[metadata_label_column]
    .astype(int)
    .to_numpy()
)


if len(df) != len(metadata):
    raise ValueError("Feature and metadata row counts do not match.")

if len(df) != len(embeddings):
    raise ValueError("Feature and embedding row counts do not match.")

if not np.array_equal(feature_urls, metadata_urls):
    raise ValueError("URL ordering mismatch between features and metadata.")

if not np.array_equal(feature_labels, metadata_labels):
    raise ValueError("Label mismatch between features and metadata.")

if not np.isfinite(embeddings).all():
    raise ValueError("Embeddings contain NaN or infinite values.")

if not set(np.unique(feature_labels)).issubset({0, 1}):
    raise ValueError("Unexpected labels. Expected only 0 and 1.")

print("Dataset alignment verified successfully!")


# ==========================================
# 7. PREPARE HANDCRAFTED FEATURES
# ==========================================

print("\nPreparing handcrafted features...")

excluded_columns = {
    url_column,
    label_column
}

feature_columns = [
    column
    for column in df.columns
    if column not in excluded_columns
]

if len(feature_columns) != 26:
    raise ValueError(
        f"Expected 26 handcrafted features, "
        f"but found {len(feature_columns)}."
    )

X_hand = (
    df[feature_columns]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0)
    .to_numpy(dtype=np.float32)
)

y = feature_labels

X_embeddings = np.asarray(
    embeddings,
    dtype=np.float32
)

if X_embeddings.shape[1] != 384:
    raise ValueError(
        f"Expected 384 embedding dimensions, "
        f"received {X_embeddings.shape[1]}."
    )

print("Handcrafted features:", X_hand.shape)
print("Transformer embeddings:", X_embeddings.shape)

print("\nLabel distribution:")

print(
    pd.Series(y)
    .value_counts()
    .sort_index()
    .rename(index={
        0: "Phishing",
        1: "Legitimate"
    })
)


# ==========================================
# 8. DOMAIN-BASED TRAIN TEST SPLIT
# ==========================================

print("\nPerforming domain-based train/test split...")

groups = np.array([
    extract_domain(url)
    for url in feature_urls
])

if len(np.unique(groups)) < 2:
    raise ValueError("Not enough unique domains for grouped splitting.")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(
        X_hand,
        y,
        groups=groups
    )
)

train_domains = set(groups[train_idx])
test_domains = set(groups[test_idx])

overlapping_domains = train_domains.intersection(test_domains)

if overlapping_domains:
    raise ValueError("Domain leakage detected between train and test.")

print("Training records:", len(train_idx))
print("Testing records:", len(test_idx))

print("Training domains:", len(train_domains))
print("Testing domains:", len(test_domains))

print("Domain overlap:", len(overlapping_domains))


# ==========================================
# 9. SCALE HANDCRAFTED FEATURES
# ==========================================

print("\nScaling handcrafted features...")

scaler = StandardScaler()

X_train_hand = scaler.fit_transform(
    X_hand[train_idx]
).astype(np.float32)

X_test_hand = scaler.transform(
    X_hand[test_idx]
).astype(np.float32)


# ==========================================
# 10. CREATE HYBRID FEATURES
# ==========================================

print("\nCombining embeddings and handcrafted features...")

X_train = np.concatenate(
    [
        X_embeddings[train_idx],
        X_train_hand
    ],
    axis=1
)

X_test = np.concatenate(
    [
        X_embeddings[test_idx],
        X_test_hand
    ],
    axis=1
)

print("Training hybrid shape:", X_train.shape)
print("Testing hybrid shape:", X_test.shape)

if X_train.shape[1] != 410:
    raise ValueError("Training features must contain 410 columns.")

if X_test.shape[1] != 410:
    raise ValueError("Testing features must contain 410 columns.")


# ==========================================
# 11. TRAIN RANDOM FOREST
# ==========================================

print("\nTraining Hybrid Random Forest...")

model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(
    X_train,
    y[train_idx]
)

print("Model training completed!")


# ==========================================
# 12. MODEL PREDICTIONS
# ==========================================

print("\nEvaluating model...")

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)

phishing_index = list(model.classes_).index(0)

phishing_probabilities = probabilities[:, phishing_index]

# Convert labels so phishing is the positive class.
y_test_phishing = (
    y[test_idx] == 0
).astype(int)

predictions_phishing = (
    predictions == 0
).astype(int)


# ==========================================
# 13. EVALUATION METRICS
# ==========================================

accuracy = accuracy_score(
    y[test_idx],
    predictions
)

phishing_precision = precision_score(
    y[test_idx],
    predictions,
    pos_label=0,
    zero_division=0
)

phishing_recall = recall_score(
    y[test_idx],
    predictions,
    pos_label=0,
    zero_division=0
)

phishing_f1 = f1_score(
    y[test_idx],
    predictions,
    pos_label=0,
    zero_division=0
)

phishing_roc_auc = roc_auc_score(
    y_test_phishing,
    phishing_probabilities
)

phishing_average_precision = average_precision_score(
    y_test_phishing,
    phishing_probabilities
)


# ==========================================
# 14. CLASSIFICATION REPORT
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========\n")

report = classification_report(
    y[test_idx],
    predictions,
    labels=[0, 1],
    target_names=["Phishing", "Legitimate"],
    zero_division=0
)

print(report)


# ==========================================
# 15. CONFUSION MATRIX
# ==========================================

matrix = confusion_matrix(
    y[test_idx],
    predictions,
    labels=[0, 1]
)

print("\n========== CONFUSION MATRIX ==========\n")

print("Rows = Actual | Columns = Predicted")
print(matrix)

tn_phishing, fp_phishing, fn_phishing, tp_phishing = (
    matrix[0, 0],
    matrix[0, 1],
    matrix[1, 0],
    matrix[1, 1]
)


# ==========================================
# 16. PRINT FINAL METRICS
# ==========================================

metrics = {

    "Accuracy": float(accuracy),

    "Phishing Precision": float(phishing_precision),

    "Phishing Recall": float(phishing_recall),

    "Phishing F1 Score": float(phishing_f1),

    "Phishing ROC AUC": float(phishing_roc_auc),

    "Phishing Average Precision": float(
        phishing_average_precision
    ),

    "Training Samples": int(len(train_idx)),

    "Testing Samples": int(len(test_idx)),

    "Training Domains": int(len(train_domains)),

    "Testing Domains": int(len(test_domains)),

    "Confusion Matrix": matrix.tolist()
}


print("\n========== FINAL EVALUATION METRICS ==========\n")

print(
    json.dumps(
        metrics,
        indent=4
    )
)


# ==========================================
# 17. FEATURE IMPORTANCE
# ==========================================

print("\n========== TOP 20 IMPORTANT FEATURES ==========\n")

all_feature_names = (
    [f"embedding_{i}" for i in range(384)]
    + list(feature_columns)
)

importances = model.feature_importances_

top_indices = np.argsort(importances)[::-1][:20]

for index in top_indices:

    print(
        all_feature_names[index],
        round(float(importances[index]), 6)
    )

embedding_importance = float(
    importances[:384].sum()
)

handcrafted_importance = float(
    importances[384:].sum()
)

print("\nEmbedding importance:", round(embedding_importance, 4))

print(
    "Handcrafted importance:",
    round(handcrafted_importance, 4)
)


# ==========================================
# 18. SAVE TRAINED MODELS
# ==========================================

print("\nSaving trained models...")

joblib.dump(
    model,
    MODEL_DIR / "hybrid_random_forest.pkl"
)

joblib.dump(
    scaler,
    MODEL_DIR / "feature_scaler.pkl"
)

# Save feature ordering for reproducibility.
joblib.dump(
    feature_columns,
    MODEL_DIR / "feature_columns.pkl"
)

with open(
    MODEL_DIR / "evaluation_metrics.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )

print("\nModels and evaluation metrics saved successfully!")

print("\nFiles saved:")

print(MODEL_DIR / "hybrid_random_forest.pkl")
print(MODEL_DIR / "feature_scaler.pkl")
print(MODEL_DIR / "feature_columns.pkl")
print(MODEL_DIR / "evaluation_metrics.json")

print("\nPHISHGUARD TRAINING COMPLETED SUCCESSFULLY!")
