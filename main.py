import logging
import sys
import joblib
from src import (
    load_data,
    preprocess_text_series,
    extract_features,
    train_naive_bayes,
    get_argument_parser,
    validate_file_paths,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

MODEL_PATH = "models/naive_bayes_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


def main() -> None:
    parser = get_argument_parser()
    args = parser.parse_args()

    try:
        validate_file_paths(args.good_file, args.promo_file)

        # Step 1: Load Data
        data = load_data(args.good_file, args.promo_file, args.nrows)
        logger.info("Data loaded.")

        # Step 2: Preprocess Text Data
        data["cleaned_text"] = preprocess_text_series(data["text"])
        logger.info("Text data preprocessed.")

        # Step 3: Extract Features
        features, vectorizer = extract_features(data["cleaned_text"])
        logger.info("Features extracted.")

        # Step 4: Train Naive Bayes Model
        model = train_naive_bayes(features, data["label"])
        logger.info("Model trained.")

        # Step 5: Save Model and Vectorizer
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        logger.info("Model and vectorizer saved.")

        logger.info("Pipeline completed.")
        logger.info("Exiting with return code 0")
        sys.exit(0)
    except FileNotFoundError as e:
        logger.error(f"An error occurred: {e}")
        logger.info("Exiting with return code 1")
        sys.exit(1)


if __name__ == "__main__":
    main()
