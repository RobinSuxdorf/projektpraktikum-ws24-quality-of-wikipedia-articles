import logging
import sys
import joblib
from src import (
    get_argument_parser,
    validate_file_paths,
    load_data,
    preprocess_text_series,
    get_vectorizer,
    train_naive_bayes,
    load_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Main function to run the data processing and model training pipeline.
    """
    parser = get_argument_parser()
    args = parser.parse_args()

    try:
        validate_file_paths(args.good_file, args.promo_file)

        logger.info(f"Arguments: {vars(args)}")

        # Load configuration
        config = load_config(args.config_path)
        logger.info(f"Configuration loaded from {args.config_path}")

        # Step 1: Load Data
        data = load_data(args.good_file, args.promo_file, args.nrows)
        logger.info(
            f"Data loaded from {args.good_file} and {args.promo_file} with nrows={args.nrows}."
        )

        # Step 2: Preprocess Text Data
        preprocessing_config = config["preprocessing"]
        data["cleaned_text"] = preprocess_text_series(
            data["text"],
            remove_stopwords=preprocessing_config["remove_stopwords"],
            apply_stemming=preprocessing_config["apply_stemming"],
            remove_numbers=preprocessing_config["remove_numbers"],
        )
        logger.info(
            f"Text data preprocessed with remove_stopwords={preprocessing_config['remove_stopwords']}, apply_stemming={preprocessing_config['apply_stemming']}, remove_numbers={preprocessing_config['remove_numbers']}."
        )

        # Step 3: Extract Features
        vectorizer_config = config["vectorizer"]
        vectorizer = get_vectorizer(
            vectorizer_config["type"],
            vectorizer_config["max_features"],
            vectorizer_config["ngram_range"],
            vectorizer_config["min_df"],
            vectorizer_config["max_df"],
        )
        features = vectorizer.fit_transform(data["cleaned_text"])
        logger.info(
            f"Features extracted using {vectorizer_config['type']} vectorizer with max_features={vectorizer_config['max_features']}, ngram_range={vectorizer_config['ngram_range']}, min_df={vectorizer_config['min_df']}, max_df={vectorizer_config['max_df']}."
        )

        # Step 4: Train Model
        model_config = config["naive_bayes"]
        model = train_naive_bayes(features, data["label"], alpha=model_config["alpha"])
        logger.info(f"Naive Bayes model trained with alpha={model_config['alpha']}.")

        # Step 5: Save Model and Vectorizer
        joblib.dump(model, args.model_path)
        joblib.dump(vectorizer, args.vectorizer_path)
        logger.info(
            f"Model saved to {args.model_path} and vectorizer saved to {args.vectorizer_path}."
        )

        logger.info("Pipeline completed.")
        logger.info("Exiting with return code 0")
        sys.exit(0)
    except FileNotFoundError as e:
        logger.error(f"An error occurred: {e}")
        logger.info("Exiting with return code 1")
        sys.exit(1)


if __name__ == "__main__":
    main()
