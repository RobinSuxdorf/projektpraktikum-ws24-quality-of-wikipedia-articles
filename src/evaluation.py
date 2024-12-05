# src/evaluation.py

import logging
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score

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

    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {accuracy:.2%}")

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    logger.info(
        f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}"
    )

    labels = [label for label in report if label.isdigit()]
    metrics = ["precision", "recall", "f1-score"]
    data = {metric: [report[label][metric] for label in labels] for metric in metrics}

    x = range(len(labels))
    bar_width = 0.2

    fig, ax = plt.subplots()

    bars1 = ax.bar(
        [xi - bar_width for xi in x], data["precision"], bar_width, label="Precision"
    )
    bars2 = ax.bar(x, data["recall"], bar_width, label="Recall")
    bars3 = ax.bar(
        [xi + bar_width for xi in x], data["f1-score"], bar_width, label="F1-Score"
    )

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    ax.set_title("Metrics per label")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()

    if accuracy:
        ax.text(
            0.5,
            0.95,
            f"Accuracy: {accuracy:.2%}",
            transform=ax.transAxes,
            ha="center",
        )

    fig.tight_layout()
    return fig