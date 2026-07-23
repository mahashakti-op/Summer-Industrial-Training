import re
import string
import time

import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------------------------------------------------------
# CONFIG - edit these small constants if your details ever change
# -------------------------------------------------------------------------
DEVELOPER_NAME = "Ishan"
MODEL_LABEL = "Linear SVM"
FEATURE_LABEL = "TF-IDF Vectorizer"
DATASET_LABEL = "Fake & Real News Dataset (Kaggle)"
BEST_MODEL_F1 = "99.3%"  # from the notebook's saved evaluation results

MODEL_PATH = "Major_Project/model.pkl"
VECTORIZER_PATH = "Major_Project/tfidf_vectorizer.pkl"


# -------------------------------------------------------------------------
# PAGE CONFIG (must be the first Streamlit command)
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Misinformation Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------------
# CUSTOM CSS - modern SaaS-dashboard look, works in both light & dark mode
# -------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit chrome for a cleaner product feel */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: transparent;}

    :root {
        --accent: #6366F1;
        --accent-dark: #4F46E5;
        --success: #10B981;
        --danger: #EF4444;
        --card-border: rgba(148, 163, 184, 0.25);
        --card-bg: rgba(148, 163, 184, 0.06);
        --muted-text: rgba(120, 130, 145, 0.95);
    }

    /* ---------- Hero ---------- */
    .hero-wrap {
        text-align: center;
        padding: 1.6rem 1rem 1.2rem 1rem;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
        background: linear-gradient(90deg, var(--accent), #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: var(--muted-text);
        max-width: 720px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* ---------- Generic card ---------- */
    .app-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.06);
        margin-bottom: 1.2rem;
    }
    .app-card h4 {
        margin-top: 0;
        font-weight: 700;
    }

    /* ---------- Prediction result cards ---------- */
    .result-card {
        border-radius: 18px;
        padding: 1.8rem 2rem;
        margin: 0.6rem 0 1.2rem 0;
        border: 1px solid var(--card-border);
        display: flex;
        align-items: center;
        gap: 1.2rem;
    }
    .result-real {
        background: rgba(16, 185, 129, 0.10);
        border-color: rgba(16, 185, 129, 0.35);
    }
    .result-fake {
        background: rgba(239, 68, 68, 0.10);
        border-color: rgba(239, 68, 68, 0.35);
    }
    .result-icon { font-size: 2.6rem; line-height: 1; }
    .result-text-real { color: var(--success); font-size: 1.7rem; font-weight: 800; margin: 0; }
    .result-text-fake { color: var(--danger); font-size: 1.7rem; font-weight: 800; margin: 0; }
    .result-sub { color: var(--muted-text); font-size: 0.95rem; margin-top: 0.2rem; }

    /* ---------- Sidebar pipeline flow ---------- */
    .pipeline-step {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 10px;
        padding: 0.5rem 0.8rem;
        text-align: center;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 2px;
    }
    .pipeline-arrow {
        text-align: center;
        color: var(--muted-text);
        margin: 0;
        line-height: 1.3;
    }

    /* ---------- Footer ---------- */
    .app-footer {
        text-align: center;
        color: var(--muted-text);
        font-size: 0.85rem;
        padding: 1.2rem 0 0.4rem 0;
        border-top: 1px solid var(--card-border);
        margin-top: 1.5rem;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------------
# NLTK setup + clean_text()
# Copied as-is from notebook.ipynb so training-time and inference-time text
# cleaning always stay perfectly identical. DO NOT change this logic here -
# change it in the notebook and retrain if it ever needs to change.
# -------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def ensure_nltk_resources():
    for resource_path, resource_name in {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet.zip": "wordnet",
        "corpora/omw-1.4.zip": "omw-1.4",
    }.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


ensure_nltk_resources()
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text):
    """Clean one title/article string for TF-IDF modelling."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    cleaned_tokens = [
        LEMMATIZER.lemmatize(word)
        for word in tokens
        if word not in STOP_WORDS and word.isalpha()
    ]
    return " ".join(cleaned_tokens)


# -------------------------------------------------------------------------
# Load the already-trained model + vectorizer (no training happens here)
# -------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error(
        "Could not find `model.pkl` and/or `tfidf_vectorizer.pkl` in this folder. "
        "Make sure both files sit next to app.py before running the app."
    )
    st.stop()


def run_prediction(raw_text):
    """Runs the full inference pipeline and times it."""
    start = time.perf_counter()
    cleaned = clean_text(raw_text)
    features = vectorizer.transform([cleaned])
    prediction = int(model.predict(features)[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = float(max(model.predict_proba(features)[0]) * 100)

    elapsed_ms = (time.perf_counter() - start) * 1000
    return prediction, confidence, elapsed_ms


# -------------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------------
if "article_input" not in st.session_state:
    st.session_state.article_input = ""
if "result" not in st.session_state:
    st.session_state.result = None


def clear_input():
    st.session_state.article_input = ""
    st.session_state.result = None


# -------------------------------------------------------------------------
# SIDEBAR - project information
# -------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 Project Information")
    st.markdown(f"**Developer**\n\n{DEVELOPER_NAME}")
    st.markdown(f"**Model Used**\n\n{MODEL_LABEL}")
    st.markdown(f"**Feature Extraction**\n\n{FEATURE_LABEL}")
    st.markdown(f"**Dataset**\n\n{DATASET_LABEL}")

    st.markdown("---")
    st.markdown("### ⚙️ Machine Learning Pipeline")
    st.markdown('<div class="pipeline-step">📝 Text Cleaning</div>', unsafe_allow_html=True)
    st.markdown('<p class="pipeline-arrow">↓</p>', unsafe_allow_html=True)
    st.markdown('<div class="pipeline-step">🔢 TF-IDF</div>', unsafe_allow_html=True)
    st.markdown('<p class="pipeline-arrow">↓</p>', unsafe_allow_html=True)
    st.markdown('<div class="pipeline-step">📐 Linear SVM</div>', unsafe_allow_html=True)
    st.markdown('<p class="pipeline-arrow">↓</p>', unsafe_allow_html=True)
    st.markdown('<div class="pipeline-step">✅ Prediction</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption(
        "This tool learns patterns from training data. It does not verify "
        "factual truth — always pair predictions with human fact-checking."
    )


# -------------------------------------------------------------------------
# HERO SECTION
# -------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-title">🤖 AI Misinformation Detector</div>
        <div class="hero-subtitle">
            Detect misleading news articles and social media posts using
            Machine Learning and Natural Language Processing.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------------
# MAIN INPUT CARD
# -------------------------------------------------------------------------

st.markdown("#### ✍️ Enter a news article or post")

st.text_area(
    label="Article text",
    key="article_input",
    height=200,
    placeholder="Paste a complete news article or social media post here...",
    label_visibility="collapsed",
)

text_is_empty = len(st.session_state.article_input.strip()) == 0

col1, col2 = st.columns([3, 1])
with col1:
    analyze_clicked = st.button(
        "🔍 Analyze",
        type="primary",
        use_container_width=True,
        disabled=text_is_empty,
    )
with col2:
    st.button("🗑 Clear", on_click=clear_input, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------------------
# RUN PREDICTION
# -------------------------------------------------------------------------
if analyze_clicked and not text_is_empty:
    with st.spinner("Analyzing text..."):
        label, confidence, elapsed_ms = run_prediction(st.session_state.article_input)
        # small delay so the spinner + card feel like a deliberate "analysis"
        time.sleep(0.3)

    st.session_state.result = {
        "label": label,
        "confidence": confidence,
        "elapsed_ms": elapsed_ms,
    }

    if label == 1:
        st.toast("Potential misinformation detected", icon="⚠️")
    else:
        st.toast("Looks like real news", icon="✅")
        st.balloons()


# -------------------------------------------------------------------------
# PREDICTION RESULT CARD
# -------------------------------------------------------------------------
if st.session_state.result is not None:
    result = st.session_state.result

    if result["confidence"] is not None:
        confidence_line = f"Confidence: {result['confidence']:.2f}%"
    else:
        confidence_line = "Confidence score is unavailable for Linear SVM."

    if result["label"] == 1:
        st.markdown(
            f"""
            <div class="result-card result-fake">
                <div class="result-icon">🚫</div>
                <div>
                    <p class="result-text-fake">FAKE NEWS</p>
                    <p class="result-sub">{confidence_line}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-card result-real">
                <div class="result-icon">✅</div>
                <div>
                    <p class="result-text-real">REAL NEWS</p>
                    <p class="result-sub">{confidence_line}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------------------------------------------------------
    # Additional information metrics
    # ---------------------------------------------------------------
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("#### 📈 Additional Information")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model Used", MODEL_LABEL)
    m2.metric("Feature Engineering", "TF-IDF")
    m3.metric("F1 Score", BEST_MODEL_F1)
    m4.metric("Prediction Time", f"{result['elapsed_ms']:.1f} ms")
    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        Built using <b>Python</b> &nbsp;•&nbsp; <b>Scikit-learn</b>
        &nbsp;•&nbsp; <b>NLTK</b> &nbsp;•&nbsp; <b>Streamlit</b>
    </div>
    """,
    unsafe_allow_html=True,
)
