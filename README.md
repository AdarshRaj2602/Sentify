# 🛍️ Amazon Review Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A comprehensive NLP research project exploring classical machine learning, word embeddings,
> deep learning (LSTM / BiLSTM), and transformer-based transfer learning for binary sentiment
> classification on Amazon product reviews.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Approaches Explored](#-approaches-explored)
- [Model Results](#-model-results)
- [Streamlit App](#-streamlit-app)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Tech Stack](#-tech-stack)

---

## 🎯 Problem Statement

Online product reviews carry enormous commercial value — they influence purchasing decisions,
surface product defects, and shape brand reputation. Manually reading thousands of reviews is
infeasible at scale.

This project automates **binary sentiment classification** — predicting whether an Amazon
product review expresses a **Positive** or **Negative** sentiment — and compares multiple NLP
approaches in terms of accuracy, speed, and interpretability.

---

## 📦 Dataset

| Property      | Detail                                      |
|---------------|---------------------------------------------|
| Source        | Amazon product reviews (electronics & accessories) |
| Size          | 1,000 labelled samples                      |
| Labels        | Positive / Negative (binary)                |
| Balance       | ~50% Positive / 50% Negative                |
| Format        | CSV (`data/raw/amazonLabelled.csv`)         |
| Columns       | `S` (serial), `Feedback` (text), `Sentiment` |

---

## 📁 Project Structure

```
Amazon_Review_Sentiment_Analysis/
│
├── app/                          # Streamlit application
│   ├── streamlit_app.py          # 🏠 Home / Dashboard
│   ├── utils.py                  # Shared loaders, inference helpers
│   └── pages/
│       ├── 1_Single_Analysis.py  # ✍️  Single review analysis
│       ├── 2_Batch_Analysis.py   # 📂 Batch file upload (CSV/XLSX/TXT)
│       ├── 3_Model_Comparison.py # 📊 Model comparison & charts
│       └── 4_About.py            # 📖 Project context & notebooks
│
├── data/
│   ├── raw/                      # Raw dataset (gitignored)
│   └── processed/                # Preprocessed CSV (gitignored)
│
├── notebooks/                    # Research notebooks
│   ├── Sentiment_Analysis_...(CountVectorizer+Logistic_reg+Random_forest+naive_bayes).ipynb
│   ├── Sentiment_Analysis_...(Word2Vec+Logistic_reg+Random_forest).ipynb
│   ├── Sentiment_Analysis_...(Model_2)(GLOVE+LSTM_BILSTM).ipynb
│   └── Sentiment_Analysis_...(Transfer_learning_Model).ipynb
│
├── outputs/
│   ├── models/                   # Trained model artifacts
│   │   ├── vectorizer.pkl        # CountVectorizer
│   │   ├── logistic.pkl          # Logistic Regression
│   │   ├── random_forest.pkl     # Random Forest
│   │   └── naive_bayes.pkl       # Naive Bayes
│   └── results.json              # Test accuracy scores
│
├── src/
│   ├── preprocessing.py          # Text cleaning pipeline
│   ├── model.py                  # Model factory helpers
│   └── train.py                  # Training + evaluation script
│
├── main.py                       # Entry point: preprocess + train
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔬 Approaches Explored

### Notebook 1 — Classical ML (CountVectorizer)
- **Vectorizer:** `CountVectorizer` with English stop-word removal
- **Models:** Logistic Regression, Random Forest, Multinomial Naive Bayes
- **Features:** Bag-of-words term frequency counts
- **EDA:** Class distribution, word clouds, top feature importance

### Notebook 2 — Word2Vec Embeddings
- **Vectorizer:** Gensim `Word2Vec` trained on the Amazon corpus
- **Embedding:** 100-dimensional dense vectors, mean-pooled per document
- **Models:** Logistic Regression, Random Forest
- **Advantage:** Captures semantic similarity (e.g., "awful" ≈ "terrible")

### Notebook 3 — GloVe + LSTM / BiLSTM
- **Embeddings:** Pre-trained GloVe 100d (Stanford)
- **Architecture:** LSTM and Bidirectional LSTM (Keras / TensorFlow)
- **Advantage:** Captures sequential context and word order
- **Limitation:** Requires more data for optimal performance

### Notebook 4 — Transfer Learning (BERT)
- **Model:** Pre-trained transformer (Hugging Face)
- **Approach:** Fine-tuning on the Amazon review corpus
- **Advantage:** State-of-the-art NLP understanding out-of-the-box
- **Limitation:** Compute-intensive; less advantage on small datasets

---

## 📊 Model Results

| Model                | Vectorizer       | Test Accuracy | Speed       |
|----------------------|------------------|:-------------:|:-----------:|
| 📊 Naive Bayes       | CountVectorizer  | **78.5%**     | ⚡ Fastest   |
| 📈 Logistic Regression| CountVectorizer | 76.5%         | ⚡ Fast      |
| 🌲 Random Forest     | CountVectorizer  | 76.0%         | 🐢 Moderate  |

> Deep learning models (LSTM, BiLSTM, BERT) are explored in notebooks — model weights are
> not serialised due to size constraints but can be retrained via the notebooks.

---

## 🌐 Streamlit App

The app provides a **5-page interactive dashboard**:

| Page | Description |
|------|-------------|
| 🏠 **Home** | Dataset overview, model accuracy cards, class distribution charts |
| ✍️ **Single Analysis** | Text input → animated per-model result cards + confidence gauges |
| 📂 **Batch Analysis** | Upload CSV / Excel / TXT → bulk predict → download results |
| 📊 **Model Comparison** | Accuracy leaderboard, radar chart, live head-to-head tester |
| 📖 **About** | Research context, notebooks, ML pipeline, tech stack |

**Design:** Dark glassmorphism theme · Purple-indigo-cyan palette · Plotly interactive charts ·
Google Fonts (Inter) · Animated sentiment badges · Confidence gauges

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Amazon_Review_Sentiment_Analysis.git
cd Amazon_Review_Sentiment_Analysis
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the models (if .pkl files are not present)

Place `amazonLabelled.csv` in `data/raw/`, then run:

```bash
python main.py
```

This will preprocess the data and train all three classical models, saving them to `outputs/models/`.

---

## 🚀 Usage

### Run the Streamlit App

```bash
cd app
streamlit run streamlit_app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Run Training Pipeline Only

```bash
python main.py
```

---

## 🛠️ Tech Stack

| Library / Tool   | Purpose                          |
|------------------|----------------------------------|
| Python 3.11      | Core language                    |
| scikit-learn     | ML models, CountVectorizer       |
| Pandas / NumPy   | Data manipulation                |
| Streamlit        | Interactive web app              |
| Plotly           | Interactive charts               |
| TensorFlow/Keras | LSTM / BiLSTM (notebooks)        |
| Gensim           | Word2Vec embeddings (notebooks)  |
| Hugging Face     | Transfer learning (notebooks)    |
| NLTK             | NLP utilities                    |
| openpyxl         | Excel file I/O in Streamlit      |
| Jupyter          | Research notebooks               |

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
    <sub>Built with ❤️ for NLP research · Amazon Review Sentiment Analysis</sub>
</div>
