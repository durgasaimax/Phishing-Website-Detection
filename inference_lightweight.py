
import re
import ipaddress
import joblib
import pandas as pd

from pathlib import Path
from urllib.parse import urlparse, unquote


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "lightweight_random_forest.pkl"
FEATURE_PATH = PROJECT_ROOT / "models" / "lightweight_feature_columns.pkl"


# ==========================================
# 2. LOAD LIGHTWEIGHT MODEL
# ==========================================

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)


# ==========================================
# 3. SUSPICIOUS URL PATTERNS
# ==========================================

SHORTENING_SERVICES = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "cutt.ly",
    "rebrand.ly",
    "shorturl.at"
}

SUSPICIOUS_WORDS = {
    "login",
    "verify",
    "account",
    "update",
    "secure",
    "banking",
    "password",
    "signin",
    "confirm",
    "wallet",
    "recover",
    "suspended",
    "authenticate"
}


# ==========================================
# 4. URL FEATURE EXTRACTION
# ==========================================

def extract_features(url):

    url = url.strip()

    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url

    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    if not hostname:
        raise ValueError("Please enter a valid website URL.")

    hostname = hostname.lower()
    decoded_url = unquote(url).lower()

    # Check IP address
    try:
        ipaddress.ip_address(hostname)
        has_ip = 1
    except ValueError:
        has_ip = 0

    # Check port
    try:
        has_port = int(parsed.port is not None)
    except ValueError:
        raise ValueError("Invalid port in URL.")

    labels = hostname.split(".")
    path = parsed.path or ""

    # Extract all 26 features
    features = {
        "url_length": len(url),
        "domain_length": len(hostname),
        "path_length": len(path),
        "query_length": len(parsed.query),
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "at_count": url.count("@"),
        "question_mark_count": url.count("?"),
        "equal_count": url.count("="),
        "ampersand_count": url.count("&"),
        "underscore_count": url.count("_"),
        "percent_count": url.count("%"),
        "digit_count": sum(c.isdigit() for c in url),
        "letter_count": sum(c.isalpha() for c in url),
        "subdomain_count_approx": max(0, len(labels) - 2),
        "hostname_label_count": len(labels),
        "is_https": int(parsed.scheme.lower() == "https"),
        "has_ip_address": has_ip,
        "has_port": has_port,
        "has_fragment": int(bool(parsed.fragment)),
        "has_query": int(bool(parsed.query)),
        "has_punycode": int("xn--" in hostname),
        "suspicious_word_count": sum(
            word in decoded_url for word in SUSPICIOUS_WORDS
        ),
        "has_www": int(hostname.startswith("www.")),
        "has_double_slash_in_path": int("//" in path),
        "has_shortening_service": int(
            hostname.removeprefix("www.") in SHORTENING_SERVICES
        )
    }

    return features


# ==========================================
# 5. PREDICT WEBSITE
# ==========================================

def predict_website(url):

    features = extract_features(url)

    # Preserve the feature names used during model training
    input_data = pd.DataFrame(
        [[features[column] for column in feature_columns]],
        columns=feature_columns
    )

    # Predict class
    prediction = int(model.predict(input_data)[0])

    # Get prediction probabilities
    probabilities = model.predict_proba(input_data)[0]

    # Class 0 = Phishing
    # Class 1 = Legitimate

    phishing_index = list(model.classes_).index(0)

    phishing_probability = float(
        probabilities[phishing_index]
    )

    legitimate_probability = 1 - phishing_probability

    return {
        "prediction": (
            "Phishing" if prediction == 0 else "Legitimate"
        ),
        "probability": round(
            phishing_probability * 100, 2
        ),
        "legitimate_probability": round(
            legitimate_probability * 100, 2
        ),
        "features": features
    }
