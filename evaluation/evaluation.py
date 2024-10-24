from typing import Any, Callable
from sklearn.metrics import classification_report


def evaluate(
    X: list[str], 
    y_true: list[int], 
    predictor: Callable[[Any], int]
) -> dict[str, dict[str, int | float]]:
    """
    Evaluate the predictor method on the dataset.

    Args:
        X (list[str]): The input data.
        y_true (list[int]): The labels.
        predictor (Callable[[Any], int]): Method for predicting the label for the data in .

    Returns:
        dict[str, dict[str, int | float]]: The classification report.
    """
    y_pred = [predictor(x) for x in X]

    return classification_report(y_true, y_pred, output_dict=True)
