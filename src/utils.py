# src/utils.py

import argparse
import os
import logging
import yaml
import joblib
import pandas as pd
from enum import Enum
from typing import Any
from src.models import Model

logger = logging.getLogger(__name__)

CONFIGS_DIR = "configs"
DATA_DIR = "data/intermediary"


class DataType(Enum):
    DATA = "data"
    FEATURES = "features"
    MODEL = "model"


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
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file {config_path} not found.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config: {e}")
        raise


def save_to_file(data: Any, filename: str) -> None:
    """
    Save data to a file based on the provided configuration.

    Args:
        data (Any): Data to be saved.
        filename (str): Name of the file to save data to.
    """
    if filename and filename.lower() != "false":
        file_path = os.path.join(DATA_DIR, filename)
        os.makedirs(DATA_DIR, exist_ok=True)

        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False)
        elif isinstance(data, Model):
            data.save(file_path)
        elif hasattr(data, "savefig"):
            data.savefig(file_path)
        else:
            joblib.dump(data, file_path)

        logger.info(f"Data saved to {file_path}.")


def load_from_file(filename: str, data_type: str) -> Any:
    """
    Load data from a file based on the provided configuration.

    Args:
        filename (str): Name of the file to load data from.
        data_type (str): Type of data being loaded (e.g., "data", "features", "model").

    Returns:
        Any: Loaded data.
    """
    if filename and filename.lower() != "false":
        file_path = os.path.join(DATA_DIR, filename)

        try:
            if data_type == DataType.DATA.value:
                return pd.read_csv(file_path)
            elif data_type == DataType.FEATURES.value:
                return joblib.load(file_path)
            elif data_type == DataType.MODEL.value:
                model = joblib.load(file_path)
                if isinstance(model, Model):
                    model.load(file_path)
                return model
            else:
                raise ValueError(f"Unknown data type: {data_type}")
        except FileNotFoundError:
            logger.error(f"File {file_path} not found for data type {data_type}.")
            raise
