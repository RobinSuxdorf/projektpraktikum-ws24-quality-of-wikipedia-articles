import logging
import sys
from src import (
    get_argument_parser,
    load_data,
    preprocess_text_series,
    get_features,
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


def run_pipeline(config: dict) -> None:
    """
    Run the data processing and model training pipeline for classification.

    Args:
        config (dict): Configuration dictionary.
    """
    usecase = config.get("usecase")
    start_step = config.get("start_step")
    load_config = config.get("load", {})
    data_file = load_config.get("data_file", "")
    features_file = load_config.get("features_file", "")
    model_file = load_config.get("model_file", "")
    if data_file:
        logger.info(f"Loading data from file: {data_file}")
        data = load_from_file(data_file, "data")
    if features_file:
        logger.info(f"Loading features from file: {features_file}")
        features = load_from_file(features_file, "features")
    if model_file:
        logger.info(f"Loading model from file: {model_file}")
        model = load_from_file(model_file, "model")

    if start_step == "data_loader":
        data_loader_config = config.get("data_loader")
        data = load_data(data_loader_config, usecase)
        save_to_file(data, data_loader_config["save"])
    else:
        logger.info("Skipping data loading step.")

    if start_step in ["data_loader", "preprocessing"]:
        preprocessing_config = config.get("preprocessing")
        data["cleaned_text"] = preprocess_text_series(
            data["text"], preprocessing_config
        )
        data = data.drop(columns=["text"])
        save_to_file(data, preprocessing_config["save"])
    else:
        logger.info("Skipping text data preprocessing step.")

    if start_step in ["data_loader", "preprocessing", "features"]:
        features_config = config.get("features")
        features = get_features(data["cleaned_text"], features_config)
        save_to_file(features, features_config["save"])
    else:
        logger.info("Skipping feature extraction step.")

    labels = data.drop(columns=["cleaned_text"])

    if start_step in ["data_loader", "preprocessing", "features", "model"]:
        model_config = config.get("model")
        model = train_model(features, labels, model_config)
        save_to_file(model, model_config["save"])
    else:
        logger.info("Skipping model training step.")

    evaluation_config = config.get("evaluation")
    figure = evaluate_model(model, features, labels)
    save_to_file(figure, evaluation_config["save"])


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
        logger.error(f"An error occurred: {e}", exc_info=True)
        logger.info("Exiting with return code 1")
        sys.exit(1)


if __name__ == "__main__":
    main()
