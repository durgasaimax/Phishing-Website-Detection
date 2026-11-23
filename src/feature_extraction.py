
import pandas as pd
import re
import ipaddress

from pathlib import Path
from urllib.parse import urlparse

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "url_dataset.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "url_features.csv"

print("=" * 60)
print("URL FEATURE EXTRACTION")
print("=" * 60)


# Extract features from one URL
def extract_features(url):

    url = str(url).strip()

    # Add a temporary scheme for parsing URLs without one
    parse_url = url if "://" in url else "//" + url

    try:
        parsed = urlparse(parse_url)
        hostname = parsed.hostname or ""
    except ValueError:
        parsed = urlparse("")
        hostname = ""

    # URL components
    domain = hostname
    path = parsed.path
    query = parsed.query

    # Check whether hostname is an IP address
    try:
        ipaddress.ip_address(hostname)
        has_ip = 1
    except ValueError:
        has_ip = 0

    # Count hostname labels
    labels = [part for part in hostname.split(".") if part]

    # Approximate subdomain count
    subdomain_count = max(0, len(labels) - 2)

    # Suspicious words
    suspicious_words = [
        "login", "verify", "account", "secure",
        "update", "banking", "signin", "password",
        "confirm", "wallet", "payment"
    ]

    lower_url = url.lower()

    suspicious_word_count = sum(
        lower_url.count(word) for word in suspicious_words
    )

    # URL features
    features = {

        "url_length": len(url),

        "domain_length": len(domain),

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

        "has_port": int(
            parsed.port is not None
        ),

        "has_fragment": int(bool(parsed.fragment)),

        "has_query": int(bool(query)),

        "has_punycode": int("xn--" in hostname.lower()),

        "suspicious_word_count": suspicious_word_count,

        "has_www": int(hostname.lower().startswith("www.")),

        "has_double_slash_in_path": int("//" in path),

        "has_shortening_service": int(
            any(
                service in hostname.lower().split(".")
                for service in ["bit", "ly", "tinyurl"]
            )
        )
    }

    return features


# Load processed dataset
df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded:", df.shape)

# Extract URL features
print("\nExtracting features. Please wait...")

feature_records = [
    extract_features(url)
    for url in df["URL"]
]

features_df = pd.DataFrame(feature_records)

# Keep original URL and target
result = pd.concat(
    [
        df[["URL", "label"]].reset_index(drop=True),
        features_df
    ],
    axis=1
)

# Save extracted features
result.to_csv(OUTPUT_FILE, index=False)

print("\nFEATURE EXTRACTION COMPLETED!")

print("\nExtracted feature count:", len(features_df.columns))

print("\nFeature names:")
print(features_df.columns.tolist())

print("\nFirst five records:")
print(result.head())

print("\nSaved file:")
print(OUTPUT_FILE)
