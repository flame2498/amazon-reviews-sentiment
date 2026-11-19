"""
Streamlit Web Application for Sentiment Analysis on Amazon Reviews
Sprint 4 Deliverable - Innomatics Research Labs NLP Project
"""

import os
import re
import html
import math
from functools import lru_cache

try:
    import numpy as np
except ImportError:
    np = None

import streamlit as st

# Configure page
st.set_page_config(
    page_title="Amazon Reviews - Sentiment & Rating Predictor",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished Amazon-like styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #131921;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .stCard {
        border-radius: 10px;
        padding: 20px;
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
    }
    .badge-pos {
        background-color: #d4edda;
        color: #155724;
        font-weight: 600;
        padding: 6px 14px;
        border-radius: 20px;
        display: inline-block;
        font-size: 1.1rem;
    }
    .badge-neu {
        background-color: #fff3cd;
        color: #856404;
        font-weight: 600;
        padding: 6px 14px;
        border-radius: 20px;
        display: inline-block;
        font-size: 1.1rem;
    }
    .badge-neg {
        background-color: #f8d7da;
        color: #721c24;
        font-weight: 600;
        padding: 6px 14px;
        border-radius: 20px;
        display: inline-block;
        font-size: 1.1rem;
    }
    .star-display {
        font-size: 1.8rem;
        color: #ff9900;
        font-weight: bold;
    }
    .metric-box {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 1. NLTK & Text Preprocessing (matching Sprint 1 & Sprint 2)
# -----------------------------------------------------------------------------
@st.cache_resource
def setup_nltk_and_lemmatizer():
    """Download NLTK corpora if needed and return stopwords + lemmatizer."""
    try:
        import nltk
        for pkg in ["stopwords", "wordnet", "omw-1.4"]:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        sw = (set(stopwords.words("english")) - {"not", "no", "nor"}) | {"br"}
        lem = WordNetLemmatizer()
        return sw, lem
    except Exception:
        # Fallback basic English stopwords if NLTK is unavailable
        basic_stop = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
            "are", "as", "at", "be", "because", "been", "before", "being", "below", "between",
            "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each",
            "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here",
            "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
            "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "of", "off",
            "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
            "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the",
            "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this",
            "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were",
            "what", "when", "where", "which", "while", "who", "whom", "why", "with", "would",
            "you", "your", "yours", "yourself", "yourselves", "br"
        }
        return basic_stop, None

STOPWORDS, _LEM = setup_nltk_and_lemmatizer()

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
HTML_RE = re.compile(r"<[^>]+>")
NEG_RE = re.compile(r"n't\b")
NONALPHA = re.compile(r"[^a-z\s]")
IRREGULAR = {
    "won't": "will not",
    "can't": "can not",
    "shan't": "shall not",
    "ain't": "is not",
    "cannot": "can not"
}

@lru_cache(maxsize=10000)
def lemmatize_word(w):
    if _LEM:
        try:
            return _LEM.lemmatize(_LEM.lemmatize(w, pos="v"), pos="n")
        except Exception:
            return w
    return w

def preprocess_text(text: str) -> str:
    """Exact preprocessing pipeline used in Sprint 1 & Sprint 2."""
    if not text:
        return ""
    text = html.unescape(str(text)).lower().replace("\u2019", "'")
    text = HTML_RE.sub(" ", URL_RE.sub(" ", text))
    for k, v in IRREGULAR.items():
        text = text.replace(k, v)
    text = NONALPHA.sub(" ", NEG_RE.sub(" not", text))
    tokens = [lemmatize_word(t) for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


# -----------------------------------------------------------------------------
# 2. Model Loader
# -----------------------------------------------------------------------------
@st.cache_resource
def load_trained_model():
    """Load best_model.joblib bundle containing vectorizer, classifier, labels."""
    candidates = [
        "best_model.joblib",
        "Text Data Modelling/best_model.joblib",
        "../best_model.joblib",
        os.path.join(os.path.dirname(__file__), "best_model.joblib")
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                import joblib
                bundle = joblib.load(p)
                return bundle, f"Loaded champion model from {os.path.basename(p)}"
            except Exception as e:
                return None, f"Found {p} but failed to load: {e}"
    return None, "Model file 'best_model.joblib' not found in search paths."

model_bundle, model_status = load_trained_model()


# -----------------------------------------------------------------------------
# 3. Prediction & Scoring Logic
# -----------------------------------------------------------------------------
def predict_sentiment_and_rating(review_text: str):
    """
    Returns:
      sentiment: 'Positive', 'Neutral', or 'Negative'
      rating_stars: int (1 to 5)
      rating_score: float (1.0 to 5.0 continuous estimation)
      probs: dict of probabilities {class: float}
      clean_text: preprocessed text
    """
    clean_text = preprocess_text(review_text)

    if model_bundle is not None:
        vectorizer = model_bundle["vectorizer"]
        model = model_bundle["model"]
        labels = list(model_bundle.get("labels", ["Negative", "Neutral", "Positive"]))

        vec = vectorizer.transform([clean_text])
        pred_label = model.predict(vec)[0]

        # Calculate calibrated probabilities from decision_function
        if hasattr(model, "decision_function"):
            df_vals = model.decision_function(vec)[0]
            if len(df_vals) == len(labels):
                # Softmax over decision function
                if np is not None:
                    exp_vals = np.exp(df_vals - np.max(df_vals))
                    probs_arr = exp_vals / np.sum(exp_vals)
                    probs = {labels[i]: float(probs_arr[i]) for i in range(len(labels))}
                else:
                    max_v = max(df_vals)
                    exp_v = [math.exp(v - max_v) for v in df_vals]
                    sum_v = sum(exp_v)
                    probs = {labels[i]: float(exp_v[i] / sum_v) for i in range(len(labels))}
            else:
                probs = {lbl: (1.0 if lbl == pred_label else 0.0) for lbl in labels}
        elif hasattr(model, "predict_proba"):
            probs_arr = model.predict_proba(vec)[0]
            probs = {labels[i]: float(probs_arr[i]) for i in range(len(labels))}
        else:
            probs = {lbl: (1.0 if lbl == pred_label else 0.0) for lbl in labels}

        # Calculate estimated star rating based on class probabilities
        # Negative -> 1-2 stars (mean 1.5), Neutral -> 3 stars, Positive -> 4-5 stars (mean 4.5)
        p_neg = probs.get("Negative", 0.0)
        p_neu = probs.get("Neutral", 0.0)
        p_pos = probs.get("Positive", 0.0)

        continuous_score = p_neg * 1.5 + p_neu * 3.0 + p_pos * 4.8
        continuous_score = max(1.0, min(5.0, continuous_score))

        if pred_label == "Positive":
            stars = 5 if probs.get("Positive", 0) > 0.65 else 4
        elif pred_label == "Negative":
            stars = 1 if probs.get("Negative", 0) > 0.65 else 2
        else:
            stars = 3

        return pred_label, stars, continuous_score, probs, clean_text

    else:
        # Fallback lexicon-based rule if model bundle is unavailable
        pos_words = {"love", "great", "best", "delicious", "good", "excellent", "favorite", "fresh", "perfect"}
        neg_words = {"terrible", "bad", "worst", "awful", "disappoint", "stale", "horrible", "waste", "not"}

        tokens = set(clean_text.split())
        pos_hits = len(tokens & pos_words)
        neg_hits = len(tokens & neg_words)

        if pos_hits > neg_hits:
            pred = "Positive"
            stars = 5
            score = 4.7
            probs = {"Positive": 0.85, "Neutral": 0.10, "Negative": 0.05}
        elif neg_hits > pos_hits:
            pred = "Negative"
            stars = 1
            score = 1.3
            probs = {"Positive": 0.05, "Neutral": 0.15, "Negative": 0.80}
        else:
            pred = "Neutral"
            stars = 3
            score = 3.0
            probs = {"Positive": 0.25, "Neutral": 0.50, "Negative": 0.25}

        return pred, stars, score, probs, clean_text


# -----------------------------------------------------------------------------
# 4. Streamlit User Interface
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🛍️ Amazon Food Reviews - Sentiment & Rating Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Sprint 4: Interactive NLP web application powered by TF-IDF & Linear Support Vector Machine</div>', unsafe_allow_html=True)

# Sidebar with Model Details & Sprint Overview
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=160)
    st.markdown("### 📋 Project Overview")
    st.info(
        "**Dataset**: Amazon Fine Food Reviews\n\n"
        "• **Total Reviews**: 568,454 (Oct 1999 – Oct 2012)\n\n"
        "• **Cleaned Dataset**: ~390k unique records\n\n"
        "• **Target**: Multi-class Sentiment (`Negative`, `Neutral`, `Positive`)"
    )

    st.markdown("### ⚙️ Champion Model")
    st.success(
        "**Linear SVM + TF-IDF Vectorizer**\n\n"
        "• **Vocabulary**: 5,000 Unigram & Bigram Features\n\n"
        "• **Macro-F1 Score**: ~0.629\n\n"
        "• **Accuracy**: ~82.7%\n\n"
        "• **Class Weight**: Balanced (handles majority bias)"
    )

    st.markdown("### 🚀 Sprints Roadmap")
    st.markdown("""
    - ✅ **Sprint 1**: EDA & Text Preprocessing
    - ✅ **Sprint 2**: Multi-Model Benchmarking (6 Algorithms)
    - ✅ **Sprint 3**: Hyperparameter Tuning & Selection
    - ✅ **Sprint 4**: Interactive Streamlit Deployment
    """)

    st.caption("Developed for Innomatics Research Labs NLP Project")

# Quick Example Selector
col_presets, col_blank = st.columns([4, 1])
with col_presets:
    st.markdown("**💡 Try Sample Reviews:**")
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)

default_text = ""
if b_col1.button("🌟 5★ Ecstatic"):
    st.session_state["review_input"] = "Absolutely love this coffee! Rich, aromatic, fresh beans and delicious taste. Will definitely buy again."
if b_col2.button("😐 3★ Mixed"):
    st.session_state["review_input"] = "The product is okay, nothing special. It tastes average and arrived on time, but not bad for the price."
if b_col3.button("💥 1★ Terrible"):
    st.session_state["review_input"] = "Terrible experience. The package arrived open and stale. Taste was awful, completely disappointed. Waste of money!"
if b_col4.button("🧹 Clear"):
    st.session_state["review_input"] = ""

# Main Input Section
user_review = st.text_area(
    "Enter Customer Review:",
    value=st.session_state.get("review_input", ""),
    height=140,
    placeholder="Type or paste an Amazon product review here... (e.g., 'The chocolate was rich and arrived perfectly packed!')"
)

col_btn, col_stats = st.columns([1, 4])
with col_btn:
    predict_clicked = st.button("🔍 Predict Sentiment", type="primary", use_container_width=True)

with col_stats:
    char_count = len(user_review)
    word_count = len(user_review.split())
    st.markdown(f"<p style='color: #666; padding-top: 8px;'>Word Count: <b>{word_count}</b> | Character Count: <b>{char_count}</b></p>", unsafe_allow_html=True)

# Prediction Output
if predict_clicked:
    if not user_review.strip():
        st.warning("⚠️ Please enter a review text first to analyze.")
    else:
        with st.spinner("Analyzing review text..."):
            sentiment, stars, continuous_score, probs, clean_text = predict_sentiment_and_rating(user_review)

        st.markdown("---")
        st.markdown("### 📊 Prediction Results")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("#### 1. Sentiment Classification")
            if sentiment == "Positive":
                st.markdown('<div class="badge-pos">🟢 POSITIVE SENTIMENT</div>', unsafe_allow_html=True)
                st.success("The customer expressed high satisfaction and positive praise.")
            elif sentiment == "Neutral":
                st.markdown('<div class="badge-neu">🟡 NEUTRAL SENTIMENT</div>', unsafe_allow_html=True)
                st.info("The customer expressed mixed or moderate sentiments.")
            else:
                st.markdown('<div class="badge-neg">🔴 NEGATIVE SENTIMENT</div>', unsafe_allow_html=True)
                st.error("The customer expressed dissatisfaction, complaints, or negative feedback.")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 2. Predicted Rating")
            star_icons = "⭐" * stars + "☆" * (5 - stars)
            st.markdown(f'<div class="star-display">{star_icons} ({stars} / 5 Stars)</div>', unsafe_allow_html=True)
            st.markdown(f"**Estimated Numerical Score**: `{continuous_score:.2f} / 5.00`")
            st.progress(float(continuous_score) / 5.0)

        with col_right:
            st.markdown("#### 3. Class Confidence Distribution")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(
                    f'<div class="metric-box"><h4>🔴 Negative</h4><h2>{probs.get("Negative", 0)*100:.1f}%</h2></div>',
                    unsafe_allow_html=True
                )
            with m2:
                st.markdown(
                    f'<div class="metric-box"><h4>🟡 Neutral</h4><h2>{probs.get("Neutral", 0)*100:.1f}%</h2></div>',
                    unsafe_allow_html=True
                )
            with m3:
                st.markdown(
                    f'<div class="metric-box"><h4>🟢 Positive</h4><h2>{probs.get("Positive", 0)*100:.1f}%</h2></div>',
                    unsafe_allow_html=True
                )

            # Horizontal bar visualization
            chart_data = {
                "Negative": probs.get("Negative", 0),
                "Neutral": probs.get("Neutral", 0),
                "Positive": probs.get("Positive", 0)
            }
            st.bar_chart(chart_data)

        # Preprocessing & Explainability Expander
        with st.expander("🔍 Preprocessing Pipeline & Feature Breakdown", expanded=False):
            st.markdown("**Original Review:**")
            st.code(user_review)
            st.markdown("**Cleaned & Lemmatized Tokens (Model Input):**")
            st.code(clean_text if clean_text else "[No informative tokens remaining]")

            tokens = clean_text.split()
            st.markdown(f"**Total Features Extracted:** `{len(tokens)} tokens`")
            if tokens:
                st.markdown("**Detected Key Terms:** " + ", ".join([f"`{t}`" for t in tokens[:15]]))

        # Model Info Footer
        st.caption(f"Status: {model_status}")

# Information section at the bottom
st.markdown("---")
with st.expander("ℹ️ How This Model Works & Technical Architecture"):
    st.markdown("""
    ### Machine Learning Workflow:
    1. **Data Cleaning**: Handled missing values, deduplication, removal of invalid helpfulness counts, and conversion of Unix timestamps.
    2. **Text Normalization**: HTML tag removal, URL stripping, contraction normalization, non-alphabetic filtering, negation preservation (`not`, `no`, `nor`), and two-pass WordNet lemmatization.
    3. **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency) using unigrams and bigrams (`ngram_range=(1,2)`) across 5,000 max features.
    4. **Model Selection**: Evaluated 6 diverse algorithms (Naive Bayes, Logistic Regression, Linear SVM, Decision Tree, Random Forest, KNN).
    5. **Evaluation Metric**: Macro-F1 was chosen over Accuracy due to the 78% positive class imbalance.
    6. **Champion Model**: Linear Support Vector Classifier (`LinearSVC`) with balanced class weights achieved the highest macro-F1 (0.629) and robust 82.7% accuracy.
    """)
