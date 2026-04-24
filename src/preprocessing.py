import pandas as pd
import re
import os

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_and_save(input_path, output_path):

    # ✅ Read correctly
    df = pd.read_csv(input_path)

    print("Raw shape:", df.shape)
    print(df.head())

    # ✅ Rename correct columns
    df = df.rename(columns={
        "Feedback": "review",
        "Sentiment": "sentiment"
    })

    # ✅ Keep only required columns
    df = df[["review", "sentiment"]]

    # ✅ Convert sentiment
    df["sentiment"] = df["sentiment"].map({
        "Positive": 1,
        "Negative": 0
    })

    # ❗ Remove nulls
    df = df.dropna()

    # ✅ Ensure text is string
    df["review"] = df["review"].astype(str)

    # ✅ Clean text
    df["review"] = df["review"].apply(clean_text)

    # ❗ Remove empty reviews
    df = df[df["review"].str.strip() != ""]

    print("Final dataset shape:", df.shape)

    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Saved to: {output_path}")