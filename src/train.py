import os
import json
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB


def train_model():
    # Load processed data
    df = pd.read_csv("data/processed/processed.csv")

    X = df["review"]
    y = df["sentiment"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Vectorization
    vectorizer = CountVectorizer(stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Models dictionary
    models = {
        "logistic": LogisticRegression(),
        "random_forest": RandomForestClassifier(),
        "naive_bayes": MultinomialNB()
    }

    # Create folder
    os.makedirs("outputs/models", exist_ok=True)

    results = {}

    # Train all models
    for name, model in models.items():
        print(f"Training {name}...")

        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)

        acc = accuracy_score(y_test, preds)
        results[name] = acc

        print(f"{name} Accuracy: {acc:.4f}")

        # Save model
        pickle.dump(model, open(f"outputs/models/{name}.pkl", "wb"))

    # Save vectorizer ONCE
    pickle.dump(vectorizer, open("outputs/models/vectorizer.pkl", "wb"))

    print("\n✅ All models trained & saved!")
    print("📊 Results:", results)

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/results.json", "w") as f:
        json.dump(results, f)