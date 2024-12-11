# src/utils.py

import argparse
import os
import logging
import yaml
import joblib
import pandas as pd
from src.models import Model

logger = logging.getLogger(__name__)


def get_argument_parser() -> argparse.ArgumentParser:
    """
    Create and return the argument parser for the script.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Load promotional and non-promotional data.",
        epilog="""Example usage: python main.py -c just-load""",
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
    config_path = os.path.join("configs", f"{config_name}.yaml")
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
        directory = "data/intermediary"
        file_path = os.path.join(directory, filename)

        os.makedirs(directory, exist_ok=True)

        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False)
        elif isinstance(data, Model):
            data.save(file_path)
        elif hasattr(data, "savefig"):
            data.savefig(file_path)
        else:
            joblib.dump(data, file_path)

        logger.info(f"Data saved to {file_path}.")


def load_from_file(filename: str, data_type: str) -> any:
    """
    Load data from a file based on the provided configuration.

    Args:
        filename (str): Name of the file to load data from.
        data_type (str): Type of data being loaded (e.g., "data", "features", "model").

    Returns:
        any: Loaded data.
    """
    if filename and filename.lower() != "false":
        directory = "data/intermediary"
        file_path = os.path.join(directory, filename)

        if data_type == "data":
            return pd.read_csv(file_path)
        elif data_type == "features":
            return joblib.load(file_path)
        elif data_type == "model":
            model = joblib.load(file_path)
            if isinstance(model, Model):
                model.load(file_path)
            return model
        else:
            raise ValueError(f"Unknown data type: {data_type}")
