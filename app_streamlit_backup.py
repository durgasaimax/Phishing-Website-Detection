
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import ipaddress

from pathlib import Path
from urllib.parse import urlparse
from sentence_transformers import SentenceTransformer
from datetime import datetime


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="PhishGuard | AI Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. PROJECT CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "hybrid_random_forest.pkl"
SCALER_PATH = ROOT / "models" / "feature_scaler.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

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


# =====================================================
# 3. CUSTOM CSS - PREMIUM DARK THEME
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #09090b;
    --panel: #111114;
    --panel-2: #18181d;
    --border: #302b3b;
    --purple: #854cff;
    --purple-2: #6d3df5;
    --yellow: #ffd400;
    --text: #f8f7fb;
    --muted: #aaa5b5;
    --green: #32d69a;
    --red: #ff6378;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
    background-color: var(--bg);
    color: var(--text);
    background-image:
      linear-gradient(rgba(255,255,255,.045) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,.045) 1px, transparent 1px);
    background-size: 66px 66px;
}
header[data-testid="stHeader"] { background: rgba(9,9,11,.92); }
.block-container { max-width: 1320px; padding-top: 1.2rem; padding-bottom: 3rem; }
section[data-testid="stSidebar"] { background: #101014; border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] * { color: var(--text); }
h1, h2, h3, h4, p, label { color: var(--text); }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -1px; }
.brand { font-family: 'Space Grotesk', sans-serif; font-size: 27px; font-weight: 700; letter-spacing: -1.2px; color: white; }
.brand span { color: var(--purple); }
.brand-caption { color: var(--muted); font-size: 10px; letter-spacing: 2.5px; margin-top: -3px; }
.topbar { border-bottom: 1px solid #29252f; padding: 0 0 14px 0; margin-bottom: 36px; }
.hero-wrap { padding: 32px 0 22px 0; }
.eyebrow { display:inline-block; color:#c5aaff; border:1px solid #49356c; background:#171020; border-radius:7px; padding:8px 13px; font-size:10px; letter-spacing:2px; text-transform:uppercase; margin-bottom:20px; }
.hero-title { font-family:'Space Grotesk',sans-serif; font-size:clamp(42px,6vw,76px); line-height:1.02; font-weight:700; letter-spacing:-3.5px; color:#fff; margin:0 0 20px 0; }
.hero-title span { color:var(--purple); }
.hero-description { color:#b7b1c2; font-size:16px; line-height:1.85; max-width:650px; }
.hero-note { color:#8f879c; font-size:12px; margin-top:20px; }
.panel, div[data-testid="stVerticalBlockBorderWrapper"] { background:rgba(20,19,25,.92); border:1px solid var(--border); border-radius:18px; }
[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:18px; }
[data-testid="stMetricLabel"] { color:var(--muted); }
[data-testid="stMetricValue"] { color:white; font-family:'Space Grotesk',sans-serif; }
.stTextInput input { background:#17151d !important; color:white !important; border:1px solid #4b405e !important; border-radius:12px !important; min-height:52px; }
.stTextInput input:focus { border-color:var(--purple) !important; box-shadow:0 0 0 2px rgba(133,76,255,.22) !important; }
.stButton > button, .stDownloadButton > button { background:var(--purple) !important; color:white !important; border:1px solid var(--purple) !important; border-radius:11px !important; min-height:48px; font-weight:700; transition:.2s; }
.stButton > button:hover, .stDownloadButton > button:hover { background:#7040e8 !important; border-color:#a17aff !important; transform:translateY(-1px); }
div[role="radiogroup"] { gap:8px; }
div[role="radiogroup"] label { background:#15131a; border:1px solid #302b3b; border-radius:10px; padding:8px 15px; }
div[role="radiogroup"] label[data-checked="true"] { background:#24163d; border-color:var(--purple); }
.result-safe { background:linear-gradient(135deg,#102820,#101b19); border:1px solid #238b68; padding:26px; border-radius:17px; text-align:center; }
.result-danger { background:linear-gradient(135deg,#32151d,#1c1117); border:1px solid #a83e55; padding:26px; border-radius:17px; text-align:center; }
.result-safe h1 { color:var(--green) !important; }
.result-danger h1 { color:var(--red) !important; }
[data-testid="stDataFrame"] { border:1px solid var(--border); border-radius:12px; }
[data-testid="stAlert"] { border-radius:12px; background:#17151d; }
.stProgress > div > div { background:var(--purple); }
hr { border-color:#302b3b; }
.footer { color:#777181; text-align:center; font-size:12px; padding:28px 0; line-height:2; }
.small-kicker { color:#a88aff; font-size:11px; letter-spacing:2px; text-transform:uppercase; }
@media (max-width: 700px) {
 .hero-title { letter-spacing:-2px; }
 .block-container { padding-left:1rem; padding-right:1rem; }
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 4. LOAD TRAINED MODELS
# =====================================================

@st.cache_resource
def load_models():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found: {SCALER_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    device = (
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    transformer = SentenceTransformer(
        MODEL_NAME,
        device=device
    )

    return model, scaler, transformer


# =====================================================
# 5. URL VALIDATION
# =====================================================

def validate_url(url):

    url = url.strip()

    if not url or len(url) > 2048:
        return False

    candidate = (
        url if "://" in url
        else "https://" + url
    )

    try:

        parsed = urlparse(candidate)

        hostname = parsed.hostname

        if parsed.scheme.lower() not in ["http", "https"]:
            return False

        if not hostname:
            return False

        if any(char.isspace() for char in url):
            return False

        _ = parsed.port

        return True

    except ValueError:
        return False


# =====================================================
# 6. HANDCRAFTED URL FEATURE EXTRACTION
# =====================================================

def extract_features(url):

    url = url.strip()

    candidate = (
        url if "://" in url
        else "https://" + url
    )

    parsed = urlparse(candidate)

    hostname = parsed.hostname or ""
    path = parsed.path
    query = parsed.query

    try:
        ipaddress.ip_address(hostname)
        has_ip = 1
    except ValueError:
        has_ip = 0

    labels = [
        part for part in hostname.split(".")
        if part
    ]

    suspicious_words = [
        "login", "verify", "account", "secure",
        "update", "banking", "signin", "password",
        "confirm", "wallet", "payment"
    ]

    lower_url = url.lower()

    suspicious_word_count = sum(
        lower_url.count(word)
        for word in suspicious_words
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

        "subdomain_count_approx": max(
            0, len(labels) - 2
        ),

        "hostname_label_count": len(labels),

        "is_https": int(
            parsed.scheme.lower() == "https"
        ),

        "has_ip_address": has_ip,

        "has_port": int(
            parsed.port is not None
        ),

        "has_fragment": int(
            bool(parsed.fragment)
        ),

        "has_query": int(
            bool(query)
        ),

        "has_punycode": int(
            "xn--" in hostname.lower()
        ),

        "suspicious_word_count": suspicious_word_count,

        "has_www": int(
            hostname.lower().startswith("www.")
        ),

        "has_double_slash_in_path": int(
            "//" in path
        ),

        "has_shortening_service": int(
            hostname.lower() in [
                "bit.ly",
                "tinyurl.com",
                "t.co",
                "is.gd"
            ]
        )
    }

    return features


# =====================================================
# 7. PREDICT WEBSITE
# =====================================================

def predict_website(
    url,
    model,
    scaler,
    transformer
):

    features = extract_features(url)

    feature_df = pd.DataFrame(
        [features]
    )

    feature_df = feature_df[FEATURE_COLUMNS]

    scaled_features = scaler.transform(
        feature_df
    ).astype(np.float32)

    embedding = transformer.encode(
        [url],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    combined_features = np.concatenate(
        [embedding, scaled_features],
        axis=1
    )

    if combined_features.shape[1] != model.n_features_in_:

        raise ValueError(
            f"Expected {model.n_features_in_} features, "
            f"received {combined_features.shape[1]}"
        )

    prediction = int(
        model.predict(combined_features)[0]
    )

    probabilities = model.predict_proba(
        combined_features
    )[0]

    phishing_index = list(
        model.classes_
    ).index(0)

    phishing_probability = float(
        probabilities[phishing_index]
    )

    return (
        prediction,
        phishing_probability,
        features
    )


# =====================================================
# 8. SESSION STATE
# =====================================================

if "scan_history" not in st.session_state:
    st.session_state.scan_history = []


# =====================================================
# 9. TOP NAVIGATION

nav_left, nav_right = st.columns([1.2, 3.8], vertical_alignment="center")
with nav_left:
    st.markdown(
        '<div class="brand">✦ PHISH<span>GUARD.</span></div>'
        '<div class="brand-caption">AI SECURITY PLATFORM</div>',
        unsafe_allow_html=True
    )
with nav_right:
    page = st.radio(
        "NAVIGATION",
        ["Home", "Scan History", "Model Performance", "About Project"],
        horizontal=True,
        label_visibility="collapsed"
    )

st.markdown('<div class="topbar"></div>', unsafe_allow_html=True)

# 10. HERO HEADER
if page == "Home":
    hero_left, hero_right = st.columns([1.35, 0.85], gap="large", vertical_alignment="center")
    with hero_left:
        st.markdown(
            """
            <div class="hero-wrap">
                <div class="eyebrow">✦ AI-POWERED WEBSITE PROTECTION</div>
                <div class="hero-title">Browse smart.<br>Stay <span>safe.</span></div>
                <div class="hero-description">
                    Detect suspicious websites before they put your information at risk.
                    Analyse URLs using transformer embeddings and machine learning.
                </div>
                <div class="hero-note">✓ URL intelligence &nbsp;&nbsp; ✓ Hybrid AI model &nbsp;&nbsp; ✓ Instant analysis</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with hero_right:
        st.markdown(
            """
            <div style="min-height:260px; display:flex; align-items:center; justify-content:center;">
              <div style="width:250px;height:250px;border:1px solid #58349a;border-radius:50%;
                   box-shadow:0 0 70px rgba(133,76,255,.18), inset 0 0 55px rgba(133,76,255,.08);
                   display:flex;align-items:center;justify-content:center;background:rgba(30,19,48,.35);">
                <div style="width:168px;height:168px;border-radius:28px;border:1px solid #7550b8;
                     background:linear-gradient(145deg,#211432,#111016);display:flex;flex-direction:column;
                     align-items:center;justify-content:center;transform:rotate(-8deg);">
                  <div style="font-size:54px;color:#9a6bff;">✦</div>
                  <div style="font-family:Space Grotesk,sans-serif;font-size:18px;font-weight:700;color:white;">PHISHGUARD</div>
                  <div style="font-size:10px;letter-spacing:2px;color:#b9a0e8;margin-top:7px;">THREAT SCANNER</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    st.markdown(
        f'<div class="hero-wrap"><div class="eyebrow">✦ PHISHGUARD SECURITY</div>'
        f'<div class="hero-title">{page}</div>'
        f'<div class="hero-description">Explore your website security analysis and project information.</div></div>',
        unsafe_allow_html=True
    )


# 11. HOME PAGE
# =====================================================

if page == "Home":

    st.markdown("## Website Security Scanner")

    st.markdown(
        """
        Analyse a website URL and identify potentially
        malicious characteristics using our trained AI model.
        """
    )

    st.info(
        "Privacy notice: URLs are analysed as text. "
        "The application does not visit the submitted website."
    )

    st.write("")

    with st.container(border=True):

        st.markdown("### Enter website URL")

        url = st.text_input(
            "Website address",
            placeholder="https://example.com/login",
            label_visibility="collapsed"
        )

        scan_button = st.button(
            "🔍  SCAN WEBSITE",
            use_container_width=True
        )

    if scan_button:

        if not validate_url(url):

            st.error(
                "Please enter a valid HTTP or HTTPS URL."
            )

        else:

            try:

                with st.spinner(
                    "Initialising AI and analysing URL..."
                ):

                    model, scaler, transformer = load_models()

                    prediction, phishing_probability, features = (
                        predict_website(
                            url.strip(),
                            model,
                            scaler,
                            transformer
                        )
                    )

                st.divider()

                st.markdown("## Analysis Result")

                if prediction == 0:

                    st.markdown(
                        """
                        <div class="result-danger">
                            <h1 style="color:#F87171;">
                                ⚠️ PHISHING DETECTED
                            </h1>
                            <p>
                                The model classified this URL
                                as potentially malicious.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    threat = "Phishing"

                else:

                    st.markdown(
                        """
                        <div class="result-safe">
                            <h1 style="color:#34D399;">
                                ✓ LEGITIMATE CLASSIFICATION
                            </h1>
                            <p>
                                The model classified this URL
                                as legitimate.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    threat = "Legitimate"

                st.write("")

                # Probability

                st.markdown("### Threat Probability")

                st.metric(
                    "Estimated phishing probability",
                    f"{phishing_probability * 100:.2f}%"
                )

                st.progress(
                    phishing_probability
                )

                st.caption(
                    "This is the model's estimated probability, "
                    "not a guarantee of website safety."
                )

                # Feature cards

                st.divider()

                st.markdown("### URL Intelligence")

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "URL Length",
                    features["url_length"]
                )

                c2.metric(
                    "Domain Length",
                    features["domain_length"]
                )

                c3.metric(
                    "Suspicious Words",
                    features["suspicious_word_count"]
                )

                c4, c5, c6 = st.columns(3)

                c4.metric(
                    "HTTPS",
                    "Enabled" if features["is_https"] else "Disabled"
                )

                c5.metric(
                    "IP Address",
                    "Detected" if features["has_ip_address"] else "Not detected"
                )

                c6.metric(
                    "Subdomains (approx.)",
                    features["subdomain_count_approx"]
                )

                # Feature table

                st.divider()

                st.markdown("### Detailed Feature Analysis")

                feature_table = pd.DataFrame(
                    list(features.items()),
                    columns=["Feature", "Value"]
                )

                st.dataframe(
                    feature_table,
                    use_container_width=True,
                    hide_index=True
                )

                # Store history

                st.session_state.scan_history.append({

                    "Date": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                    "URL": url.strip(),

                    "Result": threat,

                    "Phishing Probability": round(
                        phishing_probability * 100, 2
                    )
                })

                # Download report

                report = pd.DataFrame([{

                    "URL": url.strip(),

                    "Prediction": threat,

                    "Phishing Probability": round(
                        phishing_probability * 100, 2
                    ),

                    **features

                }])

                st.download_button(
                    "⬇ Download Scan Report",
                    data=report.to_csv(index=False),
                    file_name="phishguard_scan_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            except Exception as error:

                st.error(
                    f"Prediction error: {error}"
                )

                st.warning(
                    "Please verify the feature extraction pipeline "
                    "matches the original training implementation."
                )


# =====================================================
# 12. SCAN HISTORY
# =====================================================

elif page == "Scan History":

    st.markdown("## Scan History")

    st.caption(
        "Previous scans from this browser session."
    )

    if st.session_state.scan_history:

        history_df = pd.DataFrame(
            st.session_state.scan_history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇ Download Scan History",
            history_df.to_csv(index=False),
            file_name="phishguard_history.csv",
            mime="text/csv"
        )

        if st.button("Clear Scan History"):

            st.session_state.scan_history = []

            st.rerun()

    else:

        st.info(
            "No scans available yet. Scan a website to see your history."
        )


# =====================================================
# 13. MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.markdown("## Model Performance")

    st.caption(
        "Previously evaluated Hybrid Random Forest model."
    )

    metrics_path = (
        ROOT / "reports" / "evaluation_metrics.json"
    )

    if metrics_path.exists():

        import json

        with open(metrics_path, "r") as file:
            metrics = json.load(file)

        metric_columns = st.columns(3)

        metric_names = [
            ("Accuracy", "Accuracy"),
            ("Phishing Precision", "Phishing Precision"),
            ("Phishing Recall", "Phishing Recall"),
            ("Phishing F1 Score", "Phishing F1 Score"),
            ("Phishing ROC AUC", "ROC-AUC"),
            ("Phishing Average Precision", "Average Precision")
        ]

        for i, (key, title) in enumerate(metric_names):

            if key in metrics:

                metric_columns[i % 3].metric(
                    title,
                    f"{metrics[key] * 100:.2f}%"
                )

    else:

        st.warning(
            "Evaluation metrics file not found. "
            "Run src/evaluate.py first."
        )

    st.divider()

    st.markdown("### Evaluation Graphs")

    figure_folder = ROOT / "reports" / "figures"

    graph_names = [
        ("confusion_matrix.png", "Confusion Matrix"),
        ("roc_curve.png", "ROC Curve"),
        ("precision_recall_curve.png", "Precision-Recall Curve"),
        ("feature_importance.png", "Feature Importance")
    ]

    for filename, title in graph_names:

        graph_path = figure_folder / filename

        if graph_path.exists():

            st.markdown(f"#### {title}")

            st.image(
                str(graph_path),
                use_container_width=True
            )


# =====================================================
# 14. ABOUT PROJECT
# =====================================================

elif page == "About Project":

    st.markdown("## About PhishGuard")

    st.write(
        """
        PhishGuard is a Python-based phishing website
        detection project that uses Machine Learning
        and Transformer-based text embeddings to
        classify website URLs.
        """
    )

    st.divider()

    st.markdown("### Technology Stack")

    technologies = pd.DataFrame({

        "Technology": [
            "Python",
            "Pandas",
            "NumPy",
            "Scikit-learn",
            "Sentence Transformers",
            "PyTorch",
            "Streamlit"
        ],

        "Purpose": [
            "Core programming language",
            "Data processing",
            "Numerical operations",
            "Machine Learning",
            "URL embeddings",
            "Deep Learning framework",
            "Web application"
        ]

    })

    st.dataframe(
        technologies,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown("### Machine Learning Architecture")

    st.code(
        """
Website URL
     |
     v
URL Feature Extraction
     |
     +----------------------+
     |                      |
     v                      v
Transformer Embeddings   Handcrafted Features
     |                      |
   384 features          26 features
     |                      |
     |                  StandardScaler
     |                      |
     +----------+-----------+
                |
                v
        410 Hybrid Features
                |
                v
       Random Forest Model
                |
                v
     Phishing / Legitimate
        """,
        language="text"
    )

    st.info(
        "Dataset: PhiUSIIL Phishing URL Dataset. "
        "Model: Hybrid Random Forest classifier."
    )


# =====================================================
# 15. FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        <b>🛡️ PHISHGUARD</b>
        <br>
        AI-Powered Phishing Website Detection System
        <br>
        Built with Python • Machine Learning • Streamlit
        <br><br>
        Educational research prototype. A legitimate
        classification does not guarantee website safety.
    </div>
    """,
    unsafe_allow_html=True
)
