# src/evaluation.py

import logging
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib.category").setLevel(logging.WARNING)


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

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    logger.info(
        f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}"
    )

    metrics = {
        label: report[label]["f1-score"]
        for label in report
        if label
        not in ["accuracy", "micro avg", "macro avg", "weighted avg", "samples avg"]
    }
    fig, ax = plt.subplots()
    ax.barh(list(metrics.keys()), list(metrics.values()))
    ax.set_xlim([0, 1])
    ax.set_xlabel("F1-Score")
    ax.set_title("Model Evaluation Metrics")

    fig.tight_layout()
    return fig
