from src.preprocessing import preprocess_and_save
from src.train import train_model

if __name__ == "__main__":
    preprocess_and_save(
        "data/raw/amazonLabelled.csv",
        "data/processed/processed.csv"
    )

    train_model()