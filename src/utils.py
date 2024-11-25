# src/utils.py

import argparse
import os
import logging
import yaml
import joblib
import pandas as pd

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


def save_to_file(data, step_config: dict, step: str) -> None:
    """
    Save data to a file based on the provided configuration.

    Args:
        data: Data to be saved.
        step_config (dict): Configuration dictionary for the current step.
        step (str): Step name to determine the save path.
    """
    filename = step_config.get("save")
    if filename and filename.lower() != "false":
        if step == "data_loader" or step == "preprocessing":
            directory = "data/processed"
            file_path = f"{directory}/{filename}.csv"
        elif step == "evaluation":
            directory = "models"
            file_path = f"{directory}/{filename}.png"
        else:
            directory = "models"
            file_path = f"{directory}/{filename}.pkl"

        os.makedirs(directory, exist_ok=True)

        if step == "data_loader" or step == "preprocessing":
            data.to_csv(file_path, index=False)
        elif step == "evaluation":
            data.savefig(file_path)
        else:
            joblib.dump(data, file_path)

        logger.info(f"{step.capitalize()} data saved to {file_path}.")


# TODO: Load from correct folders
def load_from_file(file_path: str):
    """
    Load data from a specified file.

    Args:
        file_path (str): Path to the file to load data from.

    Returns:
        DataFrame: Loaded data.
    """
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".pkl"):
        return joblib.load(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .pkl")
