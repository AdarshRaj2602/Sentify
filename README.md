<div align="center">
  
  # 🛍️ Sentify: Amazon Review Sentiment Analysis
  
  **An end-to-end NLP ecosystem exploring Classical Machine Learning, Deep Neural Networks, and State-of-the-Art Transformers.**
  
  [![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.33-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
  [![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F59E0B?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/)
  [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)

  > *Online product reviews carry enormous commercial value—they influence purchasing decisions, surface product defects, and shape brand reputation. Manually reading thousands of reviews is infeasible at scale. **Sentify** automates binary sentiment classification to instantly predict whether customer feedback is **Positive** or **Negative**.*

</div>

---

## 📋 Table of Contents
1. [🎯 Problem Statement & Objective](#-problem-statement--objective)
2. [📦 Dataset Overview](#-dataset-overview)
3. [🔬 NLP Research & Approaches](#-nlp-research--approaches)
4. [🏆 Model Leaderboard](#-model-leaderboard)
5. [💻 Streamlit Application](#-streamlit-application)
6. [📁 Project Architecture](#-project-architecture)
7. [⚙️ Setup & Installation](#-setup--installation)
8. [🚀 Usage Guide (Training & Inference)](#-usage-guide-training--inference)
9. [🛠️ Full Technology Stack](#-full-technology-stack)
10. [🔮 Future Scope](#-future-scope)

---

## 🎯 Problem Statement & Objective
The core objective of this project is to build a robust sentiment classifier while systematically comparing the evolution of Natural Language Processing techniques. By tracing the path from simple word-frequency counts (Bag-of-Words) to context-aware deep learning and finally to massive, pre-trained transformer models, this project serves as both a **production-ready dashboard** and an **educational NLP playground**.

---

## 📦 Dataset Overview
The models are trained and evaluated on a labelled corpus of Amazon product reviews, specifically focusing on electronics and accessories. 

| Property | Detail |
|:---|:---|
| **Source** | Amazon Product Reviews |
| **Size** | 1,000 labelled samples (for classical training pipeline) |
| **Task** | Binary Classification (Positive / Negative) |
| **Balance** | ~50% Positive / ~50% Negative (Ensures no accuracy paradox) |
| **Data Format** | `.csv` (Found in `data/raw/amazonLabelled.csv`) |
| **Features** | `S` (Serial Number), `Feedback` (Raw Text), `Sentiment` (Target) |

---

## 🔬 NLP Research & Approaches
This project contains four in-depth Jupyter notebooks documenting the progression of NLP techniques.

### 1️⃣ Classical ML with Bag-of-Words (`CountVectorizer`)
* **Notebook:** `Sentiment_Analysis_(CountVectorizer+Logistic_reg+Random_forest+naive_bayes).ipynb`
* **Approach:** Raw text is tokenized, stripped of English stop-words, and converted into term-frequency arrays.
* **Models Tested:** Logistic Regression, Random Forest Classifier, Multinomial Naive Bayes.
* **Takeaway:** Naive Bayes performs surprisingly well due to the strong conditional independence of highly polarized words in short product reviews.

### 2️⃣ Semantic Word Embeddings (`Word2Vec`)
* **Notebook:** `Sentiment_Analysis_(Word2Vec+Logistic_reg+Random_forest).ipynb`
* **Approach:** Instead of sparse arrays, text is converted into 100-dimensional dense vectors trained on the Amazon corpus via Gensim's `Word2Vec`.
* **Advantage:** Unlike Bag-of-Words, Word2Vec understands that "awful" and "terrible" share similar semantic space, allowing the model to generalize better to unseen vocabulary.

### 3️⃣ Deep Learning with Sequence Models (`GloVe + LSTM / BiLSTM`)
* **Notebook:** `Sentiment_Analysis_(Model_2)(GLOVE+LSTM_BILSTM).ipynb`
* **Approach:** Words are mapped using pre-trained Stanford GloVe embeddings and fed sequentially into Long Short-Term Memory (LSTM) and Bidirectional LSTM networks built with Keras/TensorFlow.
* **Advantage:** Captures word order and long-term dependencies (e.g., understanding that "Not bad, actually good" is a positive statement, which simple word counts might fail to catch).

### 4️⃣ Transfer Learning with Transformers (`DistilBERT`) 🚀
* **Notebook:** `Sentiment_Analysis_(Transfer_learning_Model).ipynb`
* **Approach:** Leveraging HuggingFace's `distilbert-base-uncased-finetuned-sst-2-english`. 
* **Advantage:** Delivers state-of-the-art accuracy. DistilBERT is smaller, faster, and lighter than traditional BERT while retaining 97% of its language-understanding capabilities.

---

## 🏆 Model Leaderboard
The 4 most reliable models have been serialized and pushed to the production Streamlit web application.

| Model | Architecture | Feature Extraction | Speed | Accuracy |
|:---|:---|:---|:---:|:---:|
| **🤗 DistilBERT** | Transformer | WordPiece Tokenizer | 🐌 Moderate | **91.3%** *(SST-2)* |
| **📊 Naive Bayes** | Probabilistic | CountVectorizer | ⚡ Fastest | **78.5%** |
| **📈 Logistic Regression** | Linear Classifier | CountVectorizer | ⚡ Fast | **76.5%** |
| **🌲 Random Forest** | Ensemble Tree | CountVectorizer | 🐢 Moderate | **76.0%** |

*(Note: Deep learning LSTM/BiLSTM weights are not serialized in the repo due to GitHub file size limits, but the complete code to train and run them is available in the notebooks).*

---

## 💻 Streamlit Application
The repository includes a highly polished, production-ready frontend built with Streamlit, Vanilla CSS (Glassmorphism), and Plotly. 

**Application Pages:**
1. 🏠 **Home Dashboard:** High-level metrics, dataset class distributions, and deployed model architecture cards.
2. ✍️ **Single Review Analysis:** Input custom text, toggle between the 4 ML models, and watch animated confidence gauges render a majority verdict.
3. 📂 **Batch Analysis:** Upload bulk `.csv`, `.xlsx`, or `.txt` datasets. The app processes thousands of rows concurrently across multiple models and provides downloadable CSV/Excel reports.
4. 📊 **Model Comparison:** An interactive leaderboard featuring a multi-trace radar chart assessing models on Speed, Accuracy, Scalability, Interpretability, and Robustness.
5. 📖 **About & Notebooks:** Extensive documentation detailing the research phase and tech stack.

---

## 📁 Project Architecture
```text
Sentify/
├── app/                        # Streamlit Web Application
│   ├── streamlit_app.py        # Entry point for the dashboard
│   ├── utils.py                # Polymorphic model loaders & inference logic
│   ├── styles/                 # Dedicated CSS for Glassmorphism UI
│   └── pages/                  # Streamlit Multi-page routing
├── data/
│   ├── raw/                    # Original amazonLabelled.csv
│   └── processed/              # Cleaned datasets ready for training
├── notebooks/                  # 4 comprehensive Jupyter research environments
├── outputs/                    
│   ├── models/                 # Saved .pkl weights (LogReg, RF, Naive Bayes, Vectorizer)
│   └── results.json            # Cached evaluation metrics for the UI
├── src/                        # Training Pipeline Scripts
│   ├── preprocessing.py        # Text cleaning and tokenization
│   ├── model.py                # Model training helpers
│   └── train.py                # Evaluation and serialization logic
├── main.py                     # Root execution script to retrain classical models
├── requirements.txt            # Python dependencies
└── README.md                   # You are here!
