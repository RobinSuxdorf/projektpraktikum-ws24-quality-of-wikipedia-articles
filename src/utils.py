# utils.py

import argparse
import os
import logging

logger = logging.getLogger(__name__)


def get_argument_parser() -> argparse.ArgumentParser:
    """
    Create and return the argument parser for the script.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Load promotional and non-promotional data.",
        epilog="""
Example usage:
  Simple: python main.py
  Complex: python main.py -g data/raw/good.csv -p data/raw/promotional.csv -n 1000 -s
""",
    )
    parser.add_argument(
        "-g",
        "--good_file",
        type=str,
        default="data/raw/good.csv",
        help="Path to the CSV file containing non-promotional text data.",
    )
    parser.add_argument(
        "-p",
        "--promo_file",
        type=str,
        default="data/raw/promotional.csv",
        help="Path to the CSV file containing promotional text data.",
    )
    parser.add_argument(
        "-n",
        "--nrows",
        type=int,
        default=None,
        help="Number of rows to read from each CSV file.",
    )
    parser.add_argument(
        "-s",
        "--shuffle",
        action="store_true",
        help="Whether to shuffle the combined dataset.",
    )
    return parser


def validate_file_paths(good_file: str, promo_file: str) -> None:
    """
    Validate that the provided file paths exist.

    Args:
        good_file (str): Path to the non-promotional data file.
        promo_file (str): Path to the promotional data file.

    Raises:
        FileNotFoundError: If any of the provided file paths do not exist.
    """
    if not os.path.exists(good_file):
        logger.error(f"Good file path does not exist: {good_file}")
        raise FileNotFoundError(f"File not found: {good_file}")
    if not os.path.exists(promo_file):
        logger.error(f"Promotional file path does not exist: {promo_file}")
        raise FileNotFoundError(f"File not found: {promo_file}")
