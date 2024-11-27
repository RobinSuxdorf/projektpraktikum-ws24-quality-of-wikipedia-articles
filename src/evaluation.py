# src/evaluation.py

import logging
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


def evaluate_model(model, x_test, y_test) -> plt.Figure:
    """
    Evaluate the trained model and visualize the results.

    Args:
        model: Trained model.
        x_test: Test features.
        y_test: Test labels.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    logger.info("Evaluating the model.")
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logger.info(f"Model accuracy: {accuracy:.2f}")
    logger.info(f"Precision: {precision:.2f}")
    logger.info(f"Recall: {recall:.2f}")
    logger.info(f"F1 Score: {f1:.2f}")

    # Create a horizontal bar plot for the metrics
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    }
    fig, ax = plt.subplots()
    ax.barh(list(metrics.keys()), list(metrics.values()))
    ax.set_xlim([0, 1])
    ax.set_xlabel("Score")
    ax.set_title("Model Evaluation Metrics")

    fig.tight_layout()
    return fig
