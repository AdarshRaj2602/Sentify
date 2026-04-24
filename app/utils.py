import os
import re
import json
import pickle
import pandas as pd
import streamlit as st

# Base paths relative to app/ directory
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODELS_DIR    = os.path.join(PROJECT_ROOT, "outputs", "models")
RESULTS_PATH  = os.path.join(PROJECT_ROOT, "outputs", "results.json")
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "amazonLabelled.csv")
STYLES_DIR    = os.path.join(BASE_DIR, "styles")


def load_css(filename: str) -> None:
    """Inject a CSS file from app/styles/ into the Streamlit page."""
    path = os.path.join(STYLES_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



def clean_text(text: str) -> str:
    """Lowercase and normalise whitespace (mirrors training pipeline)."""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource(show_spinner=False)
def load_vectorizer():
    path = os.path.join(MODELS_DIR, "vectorizer.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    model_files = {
        "Logistic Regression": "logistic.pkl",
        "Random Forest":       "random_forest.pkl",
        "Naive Bayes":         "naive_bayes.pkl",
    }
    loaded = {}
    for name, fname in model_files.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            with open(path, "rb") as f:
                loaded[name] = pickle.load(f)
    return loaded


@st.cache_data(show_spinner=False)
def load_results() -> dict:
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_raw_dataset() -> pd.DataFrame:
    if os.path.exists(RAW_DATA_PATH):
        return pd.read_csv(RAW_DATA_PATH)
    return pd.DataFrame()


class TransferLearningPredictor:
    """DistilBERT fine-tuned on SST-2, wrapped for consistent inference API."""

    MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self):
        import os
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        from transformers import pipeline as hf_pipeline
        self._pipe = hf_pipeline(
            task="sentiment-analysis",
            model=self.MODEL_ID,
            framework="tf",
            truncation=True,
            max_length=512,
        )

    def predict_text(self, text: str) -> dict:
        result = self._pipe(str(text)[:1024])[0]
        label = "Positive" if result["label"] == "POSITIVE" else "Negative"
        return {
            "label":      label,
            "confidence": float(result["score"]),
            "raw_pred":   1 if label == "Positive" else 0,
        }


@st.cache_resource(show_spinner=False)
def load_transfer_model():
    """Load DistilBERT. Downloads ~250 MB on first use; returns None if unavailable."""
    try:
        return TransferLearningPredictor()
    except Exception:
        return None


def predict_sentiment(text: str, model, vectorizer=None) -> dict:
    """Run inference on one text. Handles both classical and TL models."""
    if hasattr(model, "predict_text"):
        return model.predict_text(text)

    cleaned  = clean_text(text)
    vec      = vectorizer.transform([cleaned])
    raw_pred = model.predict(vec)[0]
    label    = "Positive" if raw_pred == 1 else "Negative"

    confidence = None
    try:
        proba      = model.predict_proba(vec)
        confidence = float(proba.max())
    except AttributeError:
        pass

    return {"label": label, "confidence": confidence, "raw_pred": int(raw_pred)}


def batch_predict(df: pd.DataFrame, text_col: str, models: dict, vectorizer) -> pd.DataFrame:
    """Score every row in df with all selected models, adding a majority-vote column."""
    results_df = df.copy()
    label_cols = []

    for model_name, model in models.items():
        sentiments, confidences = [], []
        for text in df[text_col].fillna(""):
            result = predict_sentiment(str(text), model, vectorizer)
            sentiments.append(result["label"])
            confidences.append(
                round(result["confidence"] * 100, 1) if result["confidence"] else None
            )
        col_s = f"{model_name}_Sentiment"
        col_c = f"{model_name}_Confidence_%"
        results_df[col_s] = sentiments
        results_df[col_c] = confidences
        label_cols.append(col_s)

    def majority_vote(row):
        votes = [row[c] for c in label_cols]
        pos   = votes.count("Positive")
        neg   = votes.count("Negative")
        if pos > neg:
            return "Positive"
        elif neg > pos:
            return "Negative"
        return "Neutral (Tie)"

    results_df["Majority_Sentiment"] = results_df.apply(majority_vote, axis=1)
    return results_df


# Model metadata used across all pages
MODEL_META = {
    "Logistic Regression": {
        "vectorizer":  "CountVectorizer",
        "type":        "Linear Classifier",
        "speed":       "⚡ Fast",
        "icon":        "📈",
        "description": (
            "A linear model that learns a decision boundary in the TF-IDF "
            "feature space. Fast, interpretable, and solid on short texts."
        ),
    },
    "Random Forest": {
        "vectorizer":  "CountVectorizer",
        "type":        "Ensemble Tree",
        "speed":       "🐢 Moderate",
        "icon":        "🌲",
        "description": (
            "An ensemble of decision trees trained on random feature subsets. "
            "Captures non-linear patterns with minimal tuning."
        ),
    },
    "Naive Bayes": {
        "vectorizer":  "CountVectorizer",
        "type":        "Probabilistic",
        "speed":       "⚡ Fastest",
        "icon":        "📊",
        "description": (
            "Multinomial Naive Bayes on word counts. "
            "Extremely fast and surprisingly competitive for text classification."
        ),
    },
    "Transfer Learning (DistilBERT)": {
        "vectorizer":  "WordPiece Tokenizer",
        "type":        "Transformer / BERT",
        "speed":       "🐌 Slowest",
        "icon":        "🤗",
        "description": (
            "DistilBERT fine-tuned on SST-2. Retains 97% of BERT's performance "
            "at 60% the size. Captures deep contextual meaning beyond bag-of-words."
        ),
        "is_transfer": True,
    },
}

NOTEBOOK_META = [
    {
        "title": "CountVectorizer + LR + RF + Naive Bayes",
        "file":  "Sentiment_Analysis_using_Transfer_Learning_for_Amazon_Reviews_(CountVectorizer+Logistic_reg+Random_forest+naive_bayes).ipynb",
        "desc":  "Classical ML pipeline: CountVectorizer features, Logistic Regression, Random Forest, and Naive Bayes with cross-validation and confusion matrices.",
        "tags":  ["CountVectorizer", "Logistic Regression", "Random Forest", "Naive Bayes"],
    },
    {
        "title": "Word2Vec + Logistic Regression + Random Forest",
        "file":  "Sentiment_Analysis_using_Transfer_Learning_for_Amazon_Reviews_(Word2Vec+Logistic_reg+Random_forest).ipynb",
        "desc":  "Semantic word embeddings via Gensim Word2Vec trained on the Amazon corpus, fed into classical classifiers.",
        "tags":  ["Word2Vec", "Gensim", "Logistic Regression", "Random Forest"],
    },
    {
        "title": "GloVe + LSTM / BiLSTM",
        "file":  "Sentiment_Analysis_using_Transfer_Learning_for_Amazon_Reviews_(Model_2)(GLOVE+LSTM_BILSTM).ipynb",
        "desc":  "Pre-trained GloVe embeddings as input to LSTM and BiLSTM architectures in Keras/TensorFlow.",
        "tags":  ["GloVe", "LSTM", "BiLSTM", "Keras", "TensorFlow"],
    },
    {
        "title": "Transfer Learning (BERT / Transformer)",
        "file":  "Sentiment_Analysis_using_Transfer_Learning_for_Amazon_Reviews_(Transfer_learning_Model).ipynb",
        "desc":  "Fine-tuning a pre-trained transformer for binary sentiment. State-of-the-art results from large-scale pre-training.",
        "tags":  ["BERT", "Transfer Learning", "Transformers", "Hugging Face"],
    },
]


def sentiment_emoji(label: str) -> str:
    return "🟢" if label == "Positive" else "🔴"


def sentiment_color(label: str) -> str:
    return "#10B981" if label == "Positive" else "#EF4444"


ACCURACY_DISPLAY = {
    "Logistic Regression":           "logistic",
    "Random Forest":                 "random_forest",
    "Naive Bayes":                   "naive_bayes",
    "Transfer Learning (DistilBERT)": "__transfer__",
}

# DistilBERT SST-2 benchmark accuracy (not evaluated on Amazon test set)
TL_SST2_ACCURACY = 0.913
TL_MODEL_NAME    = "Transfer Learning (DistilBERT)"
