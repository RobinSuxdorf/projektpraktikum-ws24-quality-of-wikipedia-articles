# src/evaluation.py

import logging
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

logger = logging.getLogger(__name__)


def evaluate_model(model, x_test, y_test, evaluation_config: dict) -> plt.Figure:
    """
    Evaluate the trained model and visualize the results.

    Args:
        model: Trained model.
        x_test: Test features.
        y_test: Test labels.
        evaluation_config (dict): Configuration dictionary for evaluation.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    logger.info("Evaluating the model.")
    y_pred = model.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)

    return disp.figure_
