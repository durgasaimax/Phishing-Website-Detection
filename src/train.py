
import pandas as pd
import numpy as np
import joblib
import json
import ipaddress
import tldextract

from pathlib import Path
from urllib.parse import urlparse

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
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

MODEL_FOLDER.mkdir(exist_ok=True)
REPORT_FOLDER.mkdir(exist_ok=True)

print("=" * 65)
print("HYBRID PHISHING WEBSITE DETECTION - MODEL TRAINING")
print("=" * 65)


# ==========================================
# 2. LOAD DATA
# ==========================================

print("\nLoading datasets...")

df = pd.read_csv(FEATURE_FILE)
metadata = pd.read_csv(METADATA_FILE)

embeddings = np.load(EMBEDDING_FILE, mmap_mode="r")

print("Feature dataset:", df.shape)
print("Metadata:", metadata.shape)
print("Embeddings:", embeddings.shape)


# ==========================================
# 3. VERIFY DATA ALIGNMENT
# ==========================================

if not (
    len(df) == len(metadata) == embeddings.shape[0]
):
    raise ValueError("Dataset row counts do not match!")

if not df["URL"].reset_index(drop=True).equals(
    metadata["URL"].reset_index(drop=True)
):
    raise ValueError("URL order mismatch between datasets!")

if not df["label"].reset_index(drop=True).equals(
    metadata["label"].reset_index(drop=True)
):
    raise ValueError("Label mismatch between datasets!")

print("Dataset alignment verified!")


# ==========================================
# 4. PREPARE FEATURES
# ==========================================

print("\nPreparing features...")

# Labels: 0 = phishing, 1 = legitimate
y = df["label"].astype(int).to_numpy()

# Handcrafted numerical features
feature_columns = [
    column for column in df.columns
    if column not in ["URL", "label"]
]

X_handcrafted = df[feature_columns].fillna(0).to_numpy(
    dtype=np.float32
)

# Transformer embeddings
X_embeddings = embeddings

print("Handcrafted features:", X_handcrafted.shape)
print("Transformer features:", X_embeddings.shape)


# ==========================================
# 5. EXTRACT DOMAIN GROUPS
# ==========================================

print("\nExtracting domain groups...")

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

# Remove invalid/unknown domains from grouped evaluation
if np.any(groups == "unknown"):
    raise ValueError(
        "Some domains could not be extracted. "
        "Inspect these URLs before training."
    )

print("Unique domain groups:", len(np.unique(groups)))


# ==========================================
# 6. TRAIN TEST SPLIT
# ==========================================

print("\nSplitting data by domain...")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X_handcrafted, y, groups)
)

# Check both classes exist in each partition
if len(np.unique(y[train_idx])) != 2:
    raise ValueError("Training set does not contain both classes!")

if len(np.unique(y[test_idx])) != 2:
    raise ValueError("Test set does not contain both classes!")

print("Training records:", len(train_idx))
print("Testing records:", len(test_idx))

print(
    "Overlapping domains:",
    len(set(groups[train_idx]) & set(groups[test_idx]))
)


# ==========================================
# 7. MODEL EVALUATION FUNCTION
# ==========================================

results = []

def evaluate_model(name, model, X_train, X_test):

    print("\nTraining:", name)

    model.fit(X_train, y[train_idx])

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y[test_idx], predictions),
        "Precision": precision_score(
            y[test_idx], predictions, zero_division=0
        ),
        "Recall": recall_score(
            y[test_idx], predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y[test_idx], predictions, zero_division=0
        ),
        "ROC AUC": roc_auc_score(
            y[test_idx], probabilities
        )
    }

    results.append(metrics)

    print("\nPerformance:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nClassification report:")
    print(classification_report(
        y[test_idx],
        predictions,
        target_names=["Phishing", "Legitimate"],
        zero_division=0
    ))

    print("\nConfusion matrix:")
    print(confusion_matrix(y[test_idx], predictions))

    return model


# ==========================================
# 8. MODEL 1 - TF-IDF BASELINE
# ==========================================

print("\n" + "=" * 65)
print("MODEL 1: TF-IDF + LOGISTIC REGRESSION")
print("=" * 65)

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=30000,
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(
    df["URL"].iloc[train_idx]
)

X_test_tfidf = vectorizer.transform(
    df["URL"].iloc[test_idx]
)

tfidf_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

tfidf_model = evaluate_model(
    "TF-IDF + Logistic Regression",
    tfidf_model,
    X_train_tfidf,
    X_test_tfidf
)

joblib.dump(
    vectorizer,
    MODEL_FOLDER / "tfidf_vectorizer.pkl"
)

joblib.dump(
    tfidf_model,
    MODEL_FOLDER / "tfidf_model.pkl"
)


# ==========================================
# 9. MODEL 2 - TRANSFORMER EMBEDDINGS
# ==========================================

print("\n" + "=" * 65)
print("MODEL 2: TRANSFORMER + LOGISTIC REGRESSION")
print("=" * 65)

X_train_emb = np.asarray(
    embeddings[train_idx], dtype=np.float32
)

X_test_emb = np.asarray(
    embeddings[test_idx], dtype=np.float32
)

transformer_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

transformer_model = evaluate_model(
    "Transformer + Logistic Regression",
    transformer_model,
    X_train_emb,
    X_test_emb
)

joblib.dump(
    transformer_model,
    MODEL_FOLDER / "transformer_model.pkl"
)


# ==========================================
# 10. MODEL 3 - HYBRID MODEL
# ==========================================

print("\n" + "=" * 65)
print("MODEL 3: HYBRID FEATURES + LOGISTIC REGRESSION")
print("=" * 65)

scaler = StandardScaler()

X_train_hand = scaler.fit_transform(
    X_handcrafted[train_idx]
).astype(np.float32)

X_test_hand = scaler.transform(
    X_handcrafted[test_idx]
).astype(np.float32)

X_train_hybrid = np.concatenate(
    [X_train_emb, X_train_hand],
    axis=1
)

X_test_hybrid = np.concatenate(
    [X_test_emb, X_test_hand],
    axis=1
)

hybrid_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

hybrid_model = evaluate_model(
    "Hybrid + Logistic Regression",
    hybrid_model,
    X_train_hybrid,
    X_test_hybrid
)

joblib.dump(
    hybrid_model,
    MODEL_FOLDER / "hybrid_model.pkl"
)

joblib.dump(
    scaler,
    MODEL_FOLDER / "feature_scaler.pkl"
)


# ==========================================
# 11. MODEL 4 - HYBRID RANDOM FOREST
# ==========================================

print("\n" + "=" * 65)
print("MODEL 4: HYBRID + RANDOM FOREST")
print("=" * 65)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model = evaluate_model(
    "Hybrid + Random Forest",
    rf_model,
    X_train_hybrid,
    X_test_hybrid
)

joblib.dump(
    rf_model,
    MODEL_FOLDER / "hybrid_random_forest.pkl"
)


# ==========================================
# 12. SAVE RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

results_df.to_csv(
    REPORT_FOLDER / "model_comparison.csv",
    index=False
)

with open(
    REPORT_FOLDER / "training_split.json",
    "w"
) as file:

    json.dump(
        {
            "random_state": 42,
            "test_size": 0.20,
            "split_method": "GroupShuffleSplit by registrable domain",
            "train_records": int(len(train_idx)),
            "test_records": int(len(test_idx)),
            "unique_domains": int(len(np.unique(groups)))
        },
        file,
        indent=4
    )

print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

print(results_df.to_string(index=False))

print("\nTraining completed successfully!")

print("\nModels saved in:", MODEL_FOLDER)

print("Results saved in:", REPORT_FOLDER)
