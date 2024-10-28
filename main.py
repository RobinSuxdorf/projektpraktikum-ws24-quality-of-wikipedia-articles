# main.py

import logging
import joblib
from src import prepare_data, extract_features, train_naive_bayes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)

PROMOTIONAL_FILE = "data/raw/promotional.csv"
GOOD_FILE = "data/raw/good.csv"
NROWS = 100000
MODEL_PATH = "models/naive_bayes_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

if __name__ == "__main__":
    # Step 1: Load and Prepare Data
    data = prepare_data(PROMOTIONAL_FILE, GOOD_FILE, NROWS)
    logging.info("Data loaded and prepared.")

    # Step 2: Extract Features
    features, vectorizer = extract_features(data["cleaned_text"])
    logging.info("Features extracted.")

    # Step 3: Train Naive Bayes Model
    model = train_naive_bayes(features, data["label"])
    logging.info("Model trained.")

    # Step 4: Save Model and Vectorizer
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    logging.info("Model and vectorizer saved.")

    logging.info("Pipeline completed.")
