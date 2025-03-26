"""Main script for machine learning preprocessing and model training and evaluation pipeline.

Author: Sebastian Bunge
"""

import logging
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src import (
    PipelineStep,
    evaluate_model,
    get_argument_parser,
    get_features,
    load_config,
    load_data,
    load_from_file,
    preprocess_text_series,
    save_to_file,
    train_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def run_preprocessing_pipeline(config: dict) -> tuple:
    """
    Run only the data processing pipeline for classification.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        tuple: Tuple containing the features and labels.
    """
    usecase = config.get("usecase")
    start_step = PipelineStep.from_string(config.get("start_step", "data_loader"))

    data_loader_config = config.get("data_loader")
    data_file = data_loader_config["save"]
    if start_step <= PipelineStep.DATA_LOADER:
        data = load_data(data_loader_config, usecase)
        save_to_file(data, data_file)
    else:
        logger.info(f"Loading data from file: {data_file}")
        data = load_from_file(data_file, "data")

    preprocessing_config = config.get("preprocessing")
    prepocessed_data_file = preprocessing_config["save"]
    if start_step <= PipelineStep.PREPROCESSING:
        data["cleaned_text"] = preprocess_text_series(
            data["text"], preprocessing_config
        )
        data = data.drop(columns=["text"])
        save_to_file(data, prepocessed_data_file)
    else:
        logger.info(f"Loading prepocessed data from file: {prepocessed_data_file}")
        data = load_from_file(prepocessed_data_file, "data")

    features_config = config.get("features")
    features_file = features_config["save"]
    if start_step <= PipelineStep.FEATURES:
        features = get_features(data["cleaned_text"], features_config)
        save_to_file(features, features_file)
    else:
        logger.info(f"Loading features from file: {features_file}")
        features = load_from_file(features_file, "features")

    labels = data.drop(columns=["cleaned_text"])

    return features, labels


def run_model_pipeline(config: dict) -> None:
    """
    Run the model training and evaluation pipeline for classification.

    Args:
        config (dict): Configuration dictionary.
    """
    start_step = PipelineStep.from_string(config.get("start_step", "data_loader"))

    features, labels = run_preprocessing_pipeline(config)
    evaluation_config = config.get("evaluation")

    if evaluation_config.get("test_data"):
        logger.info(f"Loading test data from file: {evaluation_config['test_data']}")
        test_data = pd.read_csv(evaluation_config["test_data"])
        test_data = test_data.drop(columns=["url", "id", "title"], errors="ignore")
        preprocessing_config = config.get("preprocessing")
        test_data["cleaned_text"] = preprocess_text_series(
            test_data["text"], preprocessing_config
        )
        test_data = test_data.drop(columns=["text"])
        x_test = get_features(test_data["cleaned_text"], config.get("features"))
        y_test = test_data.drop(columns=["cleaned_text"])
        x_train = features
        y_train = labels
    else:
        model_config = config.get("model")
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=model_config.get("test_size", 0.2),
            random_state=model_config.get("random_state", None),
        )

    model_config = config.get("model")
    model_file = model_config["save"]
    if start_step <= PipelineStep.MODEL:
        model = train_model(x_train, y_train, model_config)
        save_to_file(model, model_file)
    else:
        logger.info(f"Loading model from file: {model_file}")
        model = load_from_file(model_file, "model")

    if start_step <= PipelineStep.EVALUATION:
        figure = evaluate_model(model, x_test, y_test)
        save_to_file(figure, evaluation_config["save"])


def main() -> None:
    """
    Main function to run the data processing and model training pipeline.
    """
    parser = get_argument_parser()
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        run_model_pipeline(config)
        logger.info("Pipeline completed.")
        logger.info("Exiting with return code 0")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        logger.info("Exiting with return code 1")
        sys.exit(1)


if __name__ == "__main__":
    main()
