# src/__init__.py

from .data_loader import load_data
from .data_preparation import preprocess_text_series
from .feature_engineering import extract_features
from .model import train_naive_bayes
from .utils import get_argument_parser, validate_file_paths

__all__ = [
    "load_data",
    "get_argument_parser",
    "validate_file_paths",
    "preprocess_text_series",
    "extract_features",
    "train_naive_bayes",
]
