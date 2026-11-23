
import json
import joblib
import pandas as pd

from pathlib import Path
from urllib.parse import urlparse

from sklearn.ensemble import RandomForestClassifier
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

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "url_features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "lightweight_random_forest.pkl"
FEATURE_FILE = MODEL_DIR / "lightweight_feature_columns.pkl"
METRICS_FILE = REPORT_DIR / "lightweight_model_metrics.json"

# ==========================================
# 2. FEATURE COLUMNS
# ==========================================

FEATURE_COLUMNS = [
    "url_length",
    "domain_length",
    "path_length",
    "query_length",
    "dot_count",
    "hyphen_count",
    "at_count",
    "question_mark_count",
    "equal_count",
    "ampersand_count",
    "underscore_count",
    "percent_count",
    "digit_count",
    "letter_count",
    "subdomain_count_approx",
    "hostname_label_count",
    "is_https",
    "has_ip_address",
    "has_port",
    "has_fragment",
    "has_query",
    "has_punycode",
    "suspicious_word_count",
    "has_www",
    "has_double_slash_in_path",
    "has_shortening_service"
]

TARGET_COLUMN = "label"

# ==========================================
# 3. LOAD DATASET
# ==========================================

print("Loading PhishGuard dataset...")

df = pd.read_csv(DATA_FILE)

required_columns = FEATURE_COLUMNS + [TARGET_COLUMN, "URL"]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

df = df.dropna(subset=required_columns).copy()

X = df[FEATURE_COLUMNS].astype("float32")
y = df[TARGET_COLUMN].astype(int)

print("Dataset shape:", df.shape)
print("\nLabel distribution:")
print(y.value_counts())

# ==========================================
# 4. DOMAIN-BASED TRAIN/TEST SPLIT
# ==========================================

def extract_hostname(url):
    try:
        parsed = urlparse(
            url if "://" in url else "https://" + url
        )
        return (parsed.hostname or "").lower().removeprefix("www.")
    except Exception:
        return ""

groups = df["URL"].apply(extract_hostname)

if (groups == "").any():
    raise ValueError("Some URLs have no valid hostname.")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining labels:")
print(y_train.value_counts())

print("\nTesting labels:")
print(y_test.value_counts())

if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError(
        "Both classes must be present in training and testing data."
    )

# ==========================================
# 5. TRAIN LIGHTWEIGHT RANDOM FOREST
# ==========================================

print("\nTraining lightweight Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training completed!")

# ==========================================
# 6. EVALUATION
# ==========================================

predictions = model.predict(X_test)

phishing_class_index = list(model.classes_).index(0)

phishing_probabilities = model.predict_proba(X_test)[
    :, phishing_class_index
]

metrics = {
    "Accuracy": float(accuracy_score(y_test, predictions)),
    "Phishing Precision": float(
        precision_score(y_test, predictions, pos_label=0, zero_division=0)
    ),
    "Phishing Recall": float(
        recall_score(y_test, predictions, pos_label=0, zero_division=0)
    ),
    "Phishing F1 Score": float(
        f1_score(y_test, predictions, pos_label=0, zero_division=0)
    ),
    "Phishing ROC AUC": float(
        roc_auc_score((y_test == 0).astype(int), phishing_probabilities)
    ),
    "Phishing Average Precision": float(
        average_precision_score(
            (y_test == 0).astype(int),
            phishing_probabilities
        )
    )
}

print("\n========== EVALUATION METRICS ==========")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Phishing", "Legitimate"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# ==========================================
# 7. SAVE MODEL AND FEATURE ORDER
# ==========================================

joblib.dump(model, MODEL_FILE)
joblib.dump(FEATURE_COLUMNS, FEATURE_FILE)

with open(METRICS_FILE, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\n========== SAVED FILES ==========")
print("Model:", MODEL_FILE)
print("Features:", FEATURE_FILE)
print("Metrics:", METRICS_FILE)

print("\nPhishGuard lightweight model is ready!")

