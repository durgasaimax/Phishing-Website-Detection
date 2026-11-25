import re
import ipaddress
import joblib
import pandas as pd

from pathlib import Path
from urllib.parse import urlparse


# ==========================================
# 1. PROJECT PATHS & MODEL LOADING
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "lightweight_random_forest.pkl"
FEATURE_PATH = PROJECT_ROOT / "models" / "lightweight_feature_columns.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not FEATURE_PATH.exists():
    raise FileNotFoundError(f"Feature columns not found: {FEATURE_PATH}")

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)


# ==========================================
# 2. FEATURE EXTRACTION CONSTANTS
# ==========================================

MULTI_PART_TLDS = {
    "co.uk", "gov.uk", "ac.uk", "org.uk",
    "co.in", "gov.in", "ac.in", "org.in", "net.in",
    "com.au", "net.au", "org.au", "edu.au", "gov.au",
    "co.jp", "ne.jp", "ac.jp", "go.jp",
    "com.br", "com.mx", "co.nz"
}

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
    "cutt.ly",
    "rebrand.ly",
    "tiny.cc"
]


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def get_registered_domain(hostname: str) -> str:
    """Extracts the registered domain (e.g. google.com from mail.google.com)."""
    parts = hostname.lower().split(".")
    if len(parts) >= 3 and f"{parts[-2]}.{parts[-1]}" in MULTI_PART_TLDS:
        return ".".join(parts[-3:])
    elif len(parts) >= 2:
        return ".".join(parts[-2:])
    return hostname


def extract_features(url: str) -> dict:
    """
    Extracts the 26 URL features expected by lightweight_random_forest.pkl.
    """
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    port = parsed.port
    path = parsed.path or ""
    query = parsed.query or ""

    try:
        ipaddress.ip_address(hostname)
        has_ip = 1
    except ValueError:
        has_ip = 0

    labels = [part for part in hostname.split(".") if part]
    subdomain_count = max(0, len(labels) - 2)

    lower_url = url.lower()
    suspicious_word_count = sum(lower_url.count(word) for word in SUSPICIOUS_WORDS)

    has_shortening = int(
        any(service in hostname for service in SHORTENING_SERVICES)
        or any(label in ["bit", "ly", "tinyurl"] for label in labels)
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
        "digit_count": sum(c.isdigit() for c in url),
        "letter_count": sum(c.isalpha() for c in url),
        "subdomain_count_approx": subdomain_count,
        "hostname_label_count": len(labels),
        "is_https": int(parsed.scheme.lower() == "https"),
        "has_ip_address": has_ip,
        "has_port": int(port is not None),
        "has_fragment": int(bool(parsed.fragment)),
        "has_query": int(bool(query)),
        "has_punycode": int("xn--" in hostname),
        "suspicious_word_count": suspicious_word_count,
        "has_www": int(hostname.startswith("www.")),
        "has_double_slash_in_path": int("//" in path),
        "has_shortening_service": has_shortening
    }

    return features


# ==========================================
# 4. PREDICTION FUNCTION
# ==========================================

def predict_website(url: str) -> dict:
    """
    Analyzes a website URL and classifies it as Legitimate or Phishing.
    Returns prediction (0=Phishing, 1=Legitimate), result label, probabilities, and features.
    """
    url = str(url).strip()
    if not url:
        raise ValueError("Please enter a website URL.")

    # Preserve explicit scheme if provided (http or https); default to https
    has_explicit_scheme = url.startswith(("http://", "https://"))
    if not has_explicit_scheme:
        normalized_url = "https://" + url
    else:
        normalized_url = url

    try:
        parsed = urlparse(normalized_url)
        hostname = (parsed.hostname or "").lower()
        port = parsed.port
    except Exception:
        raise ValueError("Please enter a valid website URL.")

    if not hostname:
        raise ValueError("Please enter a valid website URL.")

    is_ip = False
    try:
        ipaddress.ip_address(hostname)
        is_ip = True
    except ValueError:
        is_ip = False

    scheme = parsed.scheme.lower()

    # Match the PhiUSIIL dataset collection format:
    # 100% of legitimate domains in PhiUSIIL were registered root domains collected as www.<domain>.<tld>.
    if is_ip:
        canonical_base_url = f"{scheme}://{hostname}" + (f":{port}" if port else "")
        canonical_full_url = normalized_url
    else:
        reg_domain = get_registered_domain(hostname)
        canonical_base_url = f"{scheme}://www.{reg_domain}"

        # If apex domain without www., canonicalize host to www.<domain>
        if not hostname.startswith("www.") and hostname == reg_domain:
            canon_host = "www." + hostname
            netloc = canon_host + (f":{port}" if port else "")
            canonical_full_url = parsed._replace(netloc=netloc).geturl()
        else:
            canonical_full_url = normalized_url

    # 1. Extract full URL features
    full_feats = extract_features(canonical_full_url)
    X_full = pd.DataFrame([[full_feats[col] for col in feature_columns]], columns=feature_columns)
    phishing_index = list(model.classes_).index(0)
    p_full = float(model.predict_proba(X_full)[0][phishing_index])

    # 2. Extract registered base domain features
    base_feats = extract_features(canonical_base_url)
    X_base = pd.DataFrame([[base_feats[col] for col in feature_columns]], columns=feature_columns)
    p_base = float(model.predict_proba(X_base)[0][phishing_index])

    # 3. Security evaluation & dataset bias mitigation:
    # - Strict red flags: IP address, Punycode, shortening service
    if is_ip or full_feats["has_punycode"] or full_feats["has_shortening_service"]:
        final_prob = max(p_full, 0.98)
    # - If base domain is classified as phishing by the model:
    elif p_base >= 0.50:
        final_prob = max(p_full, p_base)
    # - If base domain is legitimate (p_base < 0.20):
    elif p_base < 0.20:
        suspicious_words = full_feats["suspicious_word_count"]
        has_subdomain = full_feats["subdomain_count_approx"] > 1 and not hostname.startswith("www.")

        if not full_feats["is_https"]:
            final_prob = max(p_full, 0.85)
        elif has_subdomain and suspicious_words > 0:
            final_prob = max(p_full, 0.90)
        else:
            # Safe legitimate domain (mitigates dataset zero-path bias)
            risk_adjustment = min(0.15, suspicious_words * 0.05)
            final_prob = min(0.35, p_base + risk_adjustment)
    else:
        final_prob = max(p_full, p_base)

    final_prob = min(1.0, max(0.0, final_prob))

    # Prediction convention: 0 = Phishing, 1 = Legitimate
    # JavaScript in static/js/script.js line 93: if (data.prediction === 0)
    prediction = 0 if final_prob >= 0.50 else 1
    result = "Phishing" if prediction == 0 else "Legitimate"

    phishing_percentage = round(final_prob * 100, 2)
    legitimate_percentage = round((1.0 - final_prob) * 100, 2)

    return {
        "url": normalized_url,
        "prediction": prediction,
        "result": result,
        "probability": phishing_percentage,
        "legitimate_probability": legitimate_percentage,
        "features": full_feats
    }
