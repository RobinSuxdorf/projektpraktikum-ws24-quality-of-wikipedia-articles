import logging
import sys
from src import (
    get_argument_parser,
    load_data,
    preprocess_text_series,
    get_vectorizer,
    train_model,
    evaluate_model,
    load_config,
    save_to_file,
    load_from_file,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def run_pipeline(config) -> None:
    """
    Run the data processing and model training pipeline for binary good-promotional classification.

    Args:
        config (dict): Configuration dictionary.
    """
    usecase = config.get("usecase")
    start_step = config.get("start_step")
    load_config = config.get("load")
    data_file = load_config.get("data_file")
    features_file = load_config.get("features_file")
    model_file = load_config.get("model_file")
    if data_file:
        logger.info(f"Loading data from file: {data_file}")
        data = load_from_file(load_config["data_file"])
    if features_file:
        logger.info(f"Loading features from file: {features_file}")
        features = load_from_file(load_config["features_file"])
    if model_file:
        logger.info(f"Loading model from file: {model_file}")
        model = load_from_file(load_config["model_file"])

    if start_step == "data_loader":
        logger.info("Starting data loading step.")
        data_loader_config = config.get("data_loader")
        data = load_data(data_loader_config, usecase)
        logger.info(f"First few rows of the loaded data:\n{data.head()}")
        save_to_file(data, data_loader_config)
    else:
        logger.info("Skipping data loading step.")

    if start_step in ["data_loader", "preprocessing"]:
        logger.info("Starting text data preprocessing step.")
        preprocessing_config = config.get("preprocessing")
        data["cleaned_text"] = preprocess_text_series(
            data["text"], preprocessing_config
        )
        logger.info(f"Text data preprocessed with {preprocessing_config}")
        data.drop(columns=["text"], inplace=True)
        save_to_file(data, preprocessing_config)
    else:
        logger.info("Skipping text data preprocessing step.")

    if start_step in ["data_loader", "preprocessing", "vectorizer"]:
        logger.info("Starting feature extraction step.")
        vectorizer_config = config.get("vectorizer")
        vectorizer = get_vectorizer(vectorizer_config)
        features = vectorizer.fit_transform(data["cleaned_text"])
        logger.info(f"Features extracted with {vectorizer_config}")
        save_to_file(features, vectorizer_config)
    else:
        logger.info("Skipping feature extraction step.")

    if start_step in ["data_loader", "preprocessing", "vectorizer", "model"]:
        logger.info("Starting model training step.")
        model_config = config.get("model")
        # TODO: enable multilabel
        model = train_model(features, data["label"], model_config)
        logger.info(f"Model trained with {model_config}.")
        save_to_file(model, model_config)
    else:
        logger.info("Skipping model training step.")

    logger.info("Starting model evaluation step.")
    evaluation_config = config.get("evaluation")
    # TODO: enable multilabel
    figure = evaluate_model(model, features, data["label"])
    logger.info("Figure created.")
    save_to_file(figure, evaluation_config)


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
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.info("Exiting with return code 1")
        sys.exit(1)


if __name__ == "__main__":
    main()
