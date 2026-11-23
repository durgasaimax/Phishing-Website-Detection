
import ipaddress
import joblib
import numpy as np

from pathlib import Path
from urllib.parse import urlparse
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "hybrid_random_forest.pkl"
SCALER_PATH = ROOT / "models" / "feature_scaler.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ==========================================
# 2. FEATURE CONFIGURATION
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

SUSPICIOUS_WORDS = [
    "login",
    "verify",
    "account",
    "secure",
    "update",
    "banking",
    "signin",
    "password",
    "confirm",
    "wallet",
    "payment"
]

SHORTENING_SERVICES = [
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "rebrand.ly",
    "cutt.ly"
]


# ==========================================
# 3. LOAD MODELS
# ==========================================

print("Loading PhishGuard models...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not SCALER_PATH.exists():
    raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

transformer = SentenceTransformer(
    MODEL_NAME,
    device="cpu"
)

print("All models loaded successfully!")


# ==========================================
# 4. URL VALIDATION
# ==========================================

def validate_url(url):

    if not isinstance(url, str):
        raise ValueError("Please enter a valid URL.")

    url = url.strip()

    if not url:
        raise ValueError("URL cannot be empty.")

    if len(url) > 2048:
        raise ValueError("URL is too long.")

    if any(char.isspace() for char in url):
        raise ValueError("URL must not contain spaces.")

    if not url.startswith(("http://", "https://")):
        raise ValueError(
            "Please enter a URL beginning with http:// or https://"
        )

    parsed = urlparse(url)

    if not parsed.hostname:
        raise ValueError("Invalid URL. Please enter a valid hostname.")

    try:
        parsed.port
    except ValueError:
        raise ValueError("Invalid port number in URL.")

    return url


# ==========================================
# 5. HANDCRAFTED FEATURE EXTRACTION
# ==========================================

def extract_features(url):

    url = str(url).strip()

    parse_url = (
        url if "://" in url
        else "//" + url
    )

    parsed = urlparse(parse_url)

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    try:
        ipaddress.ip_address(hostname)
        has_ip = 1
    except ValueError:
        has_ip = 0

    labels = [
        label for label in hostname.split(".")
        if label
    ]

    subdomain_count = max(
        0,
        len(labels) - 2
    )

    suspicious_word_count = sum(
        word in url.lower()
        for word in SUSPICIOUS_WORDS
    )

    features = {

        "url_length": len(url),

        "domain_length": len(hostname),

        "path_length": len(path),

        "query_length": len(query),

        "dot_count": url.count("."),

        "hyphen_count": url.count("-"),

        "at_count": url.count("@"),

        "question_mark_count": url.count("?"),

        "equal_count": url.count("="),

        "ampersand_count": url.count("&"),

        "underscore_count": url.count("_"),

        "percent_count": url.count("%"),

        "digit_count": sum(
            char.isdigit() for char in url
        ),

        "letter_count": sum(
            char.isalpha() for char in url
        ),

        "subdomain_count_approx": subdomain_count,

        "hostname_label_count": len(labels),

        "is_https": int(parsed.scheme == "https"),

        "has_ip_address": has_ip,

        "has_port": int(parsed.port is not None),

        "has_fragment": int(bool(parsed.fragment)),

        "has_query": int(bool(query)),

        "has_punycode": int("xn--" in hostname.lower()),

        "suspicious_word_count": suspicious_word_count,

        "has_www": int(hostname.lower().startswith("www.")),

        "has_double_slash_in_path": int("//" in path),

        "has_shortening_service": int(
            any(
                hostname.lower() == service
                or hostname.lower().endswith("." + service)
                for service in SHORTENING_SERVICES
            )
        )
    }

    return features


# ==========================================
# 6. WEBSITE PREDICTION
# ==========================================

def predict_website(url):

    url = validate_url(url)

    # Extract handcrafted features
    features = extract_features(url)

    # Convert features to NumPy array
    # This also fixes the StandardScaler warning

    feature_array = np.array(
        [[features[col] for col in FEATURE_COLUMNS]],
        dtype=np.float32
    )

    # Scale handcrafted features
    scaled_features = scaler.transform(
        feature_array
    ).astype(np.float32)

    # Generate transformer embeddings
    embeddings = transformer.encode(
        [url],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype(np.float32)

    # Combine embeddings and handcrafted features
    hybrid_features = np.concatenate(
        [embeddings, scaled_features],
        axis=1
    )

    # Verify model input dimensions
    if hybrid_features.shape[1] != model.n_features_in_:
        raise ValueError(
            f"Feature mismatch: model expects "
            f"{model.n_features_in_} features, "
            f"but received {hybrid_features.shape[1]}."
        )

    # Generate prediction
    prediction = int(model.predict(hybrid_features)[0])

    # Obtain class probabilities
    probabilities = model.predict_proba(
        hybrid_features
    )[0]

    # Label mapping: 0 = Phishing, 1 = Legitimate
    phishing_index = list(model.classes_).index(0)

    phishing_probability = float(
        probabilities[phishing_index]
    )

    result = (
        "Phishing"
        if prediction == 0
        else "Legitimate"
    )

    return {

        "url": url,

        "prediction": prediction,

        "result": result,

        "phishing_probability": phishing_probability,

        "features": features
    }
