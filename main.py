import logging
import sys
from src import (
    get_argument_parser,
    validate_file_paths,
    load_data,
    preprocess_text_series,
    get_vectorizer,
    train_naive_bayes,
    evaluate_model,
    load_config,
    save_to_file,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def run_pipeline(config) -> None:
    """
    Run the data processing and model training pipeline.

    Args:
        config (dict): Configuration dictionary.
    """
    logger.info("Starting data loading step.")
    data_loader_config = config.get("data_loader")
    if data_loader_config:
        validate_file_paths(data_loader_config)
        data = load_data(data_loader_config)
        logger.info(f"First few rows of the loaded data:\n{data.head()}")
        save_to_file(data, data_loader_config, "data_loader")
    else:
        logger.warning("Data loading step is not defined in the configuration.")
        return

    logger.info("Starting text data preprocessing step.")
    preprocessing_config = config.get("preprocessing")
    if preprocessing_config:
        data["cleaned_text"] = preprocess_text_series(
            data["text"], preprocessing_config
        )
        logger.info(
            f"Text data preprocessed with remove_stopwords={preprocessing_config['remove_stopwords']}, apply_stemming={preprocessing_config['apply_stemming']}, remove_numbers={preprocessing_config['remove_numbers']}."
        )
        data.drop(columns=["text"], inplace=True)
        save_to_file(data, preprocessing_config, "preprocessing")
    else:
        logger.warning("Preprocessing step is not defined in the configuration.")
        return

    logger.info("Starting feature extraction step.")
    vectorizer_config = config.get("vectorizer")
    if vectorizer_config:
        vectorizer = get_vectorizer(vectorizer_config)
        features = vectorizer.fit_transform(data["cleaned_text"])
        logger.info(
            f"Features extracted using {vectorizer_config['type']} vectorizer with max_features={vectorizer_config['max_features']}, ngram_range={vectorizer_config['ngram_range']}, min_df={vectorizer_config['min_df']}, max_df={vectorizer_config['max_df']}."
        )
        save_to_file(vectorizer, vectorizer_config, "vectorizer")
    else:
        logger.warning("Vectorizer step is not defined in the configuration.")
        return

    logger.info("Starting model training step.")
    model_config = config.get("naive_bayes")
    if model_config:
        model = train_naive_bayes(features, data["label"], model_config)
        logger.info(f"Naive Bayes model trained with alpha={model_config['alpha']}.")
        save_to_file(model, model_config, "naive_bayes")
    else:
        logger.warning("Model training step is not defined in the configuration.")
        return

    logger.info("Starting model evaluation step.")
    evaluation_config = config.get("evaluation")
    if evaluation_config:
        figure = evaluate_model(model, features, data["label"], evaluation_config)
        logger.info("Figure created.")
        save_to_file(figure, evaluation_config, "evaluation")
    else:
        logger.warning("Evaluation step is not defined in the configuration.")


def main() -> None:
    """
    Main function to run the data processing and model training pipeline.
    """
    parser = get_argument_parser()
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        run_pipeline(config)
        logger.info("Pipeline completed.")
        logger.info("Exiting with return code 0")
        sys.exit(0)
    except FileNotFoundError as e:
        logger.error(f"An error occurred: {e}")
        logger.info("Exiting with return code 1")
        sys.exit(1)


if __name__ == "__main__":
    main()
