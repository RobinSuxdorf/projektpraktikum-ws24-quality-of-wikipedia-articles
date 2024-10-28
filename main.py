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
NROWS = None # None = all rows
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

# EXAMPLE FULL RUN
# 2024-10-28 16:33:42,582 - root - INFO - Loading promotional and non-promotional data.
# 2024-10-28 16:33:45,806 - root - INFO - Preprocessing promotional data.
# 2024-10-28 16:33:45,806 - root - INFO - Preprocessing text data.
# 2024-10-28 16:33:49,540 - root - INFO - Preprocessing non-promotional data.
# 2024-10-28 16:33:49,540 - root - INFO - Preprocessing text data.
# 2024-10-28 16:34:04,391 - root - INFO - Combining datasets.
# 2024-10-28 16:34:04,396 - root - INFO - Data preparation complete.
# 2024-10-28 16:34:04,396 - root - INFO - Data loaded and prepared.
# 2024-10-28 16:34:04,396 - root - INFO - Extracting features from text data using TF-IDF Vectorizer.
# 2024-10-28 16:34:24,408 - root - INFO - Feature extraction complete.
# 2024-10-28 16:34:24,408 - root - INFO - Features extracted.
# 2024-10-28 16:34:24,408 - root - INFO - Splitting data into training and testing sets.
# 2024-10-28 16:34:24,430 - root - INFO - Training the Naive Bayes model.
# 2024-10-28 16:34:24,445 - root - INFO - Making predictions on the test set.
# 2024-10-28 16:34:24,456 - root - INFO - Model accuracy: 0.88
# 2024-10-28 16:34:24,456 - root - INFO - Precision: 0.89
# 2024-10-28 16:34:24,456 - root - INFO - Recall: 0.83
# 2024-10-28 16:34:24,456 - root - INFO - F1 Score: 0.86
# 2024-10-28 16:34:24,456 - root - INFO - Model trained.
# 2024-10-28 16:34:24,466 - root - INFO - Model and vectorizer saved.
# 2024-10-28 16:34:24,466 - root - INFO - Pipeline completed.
