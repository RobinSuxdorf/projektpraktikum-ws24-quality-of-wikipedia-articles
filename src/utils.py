# src/utils.py

import argparse
import logging
import os
from enum import IntEnum, StrEnum
import joblib
import pandas as pd
import yaml
from src.models import Model
from src.vectorizer import Vectorizer

logger = logging.getLogger(__name__)

CONFIGS_DIR = "configs"
DATA_DIR = "data/intermediary"


class DataType(StrEnum):
    DATA = "data"
    FEATURES = "features"
    MODEL = "model"


class PipelineStep(IntEnum):
    DATA_LOADER = 0
    PREPROCESSING = 1
    FEATURES = 2
    MODEL = 3
    EVALUATION = 4

    def from_string(step_name: str):
        step_name = step_name.upper()
        if step_name == "DATA_LOADER":
            return PipelineStep.DATA_LOADER
        elif step_name == "PREPROCESSING":
            return PipelineStep.PREPROCESSING
        elif step_name == "FEATURES":
            return PipelineStep.FEATURES
        elif step_name == "MODEL":
            return PipelineStep.MODEL
        elif step_name == "EVALUATION":
            return PipelineStep.EVALUATION


def get_argument_parser() -> argparse.ArgumentParser:
    """
    Create and return the argument parser for the script.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Load promotional and non-promotional data.",
        epilog="Example usage: python main.py -c just-load",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Name of the YAML configuration file.",
    )
    return parser


def load_config(config_name: str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_name (str): Name of the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    config_path = os.path.join(CONFIGS_DIR, f"{config_name}.yaml")
    logger.info(f"Loading config from {config_path}.")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def save_to_file(data: any, filename: str) -> None:
    """
    Save data to a file based on the provided configuration.

    Args:
        data (any): Data to be saved.
        filename (str): Name of the file to save data to.
    """
    if filename and filename.lower() != "false":
        file_path = os.path.join(DATA_DIR, filename)
        os.makedirs(DATA_DIR, exist_ok=True)

        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False)
        elif isinstance(data, Model) or isinstance(data, Vectorizer):
            data.save(file_path)
        elif hasattr(data, "savefig"):
            data.savefig(file_path)
        else:
            logger.error(f"Unsupported data type for saving: {type(data)}")

        logger.info(f"Data saved to {file_path}.")


def load_from_file(filename: str, data_type: str) -> any:
    """
    Load data from a file based on the provided configuration.

    Args:
        filename (str): Name of the file to load data from.
        data_type (str): Type of data being loaded.

    Returns:
        any: Loaded data.
    """
    if filename and filename.lower() != "false":
        file_path = os.path.join(DATA_DIR, filename)

        if data_type == DataType.DATA:
            return pd.read_csv(file_path)
        elif data_type in [DataType.FEATURES, DataType.MODEL]:
            data = joblib.load(file_path)
            if isinstance(data, Model) or isinstance(data, Vectorizer):
                data.load(file_path)
            return data
        else:
            logger.error(
                f"Invalid data type '{data_type}'. Supported types: {[dt for dt in DataType]}."
            )
