import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class ModelPerformanceVisualizer:
    def __init__(self):
        """Initialize the visualizer with default style settings."""
        sns.set_theme(style="whitegrid")
        self.colors = sns.color_palette("muted", 5)
        self.fig_size = (12, 8)

    def set_data(self, data: dict):
        """
        Set the performance data for visualization.

        Args:
            data (dict): A dictionary containing model performance data.
        """
        rows = []
        for model, usecases in data.items():
            for usecase, recall in usecases.items():
                rows.append({"model": model, "usecase": usecase, "recall": recall})
        self.data = pd.DataFrame(rows)

    def bar_plot(self, save_fig=True, show_fig=True):
        """
        Generate and display a bar plot of model performance.

        Args:
            save_fig (bool, optional): Whether to save the figure to a file. Defaults to True.
            show_fig (bool, optional): Whether to display the figure. Defaults to True.
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.fig_size)

        # Create the grouped bar chart using standard barplot
        bars = sns.barplot(
            data=self.data,
            x="usecase",
            y="recall",
            hue="model",
            palette=self.colors,
            ax=ax,
        )

        # Customize the plot
        ax.set_title("Model Performance Comparison (Macro Avg Recall)", fontsize=16)
        ax.set_xlabel("Use Case", fontsize=14)
        ax.set_ylabel("Macro Average Recall", fontsize=14)
        ax.set_ylim(0, 1.0)

        # Add value labels at the bottom of bars with white text
        for i, container in enumerate(bars.containers):
            for j, bar in enumerate(container):
                value = self.data[
                    (self.data["model"] == self.data["model"].unique()[i])
                    & (self.data["usecase"] == self.data["usecase"].unique()[j])
                ]["recall"].values[0]
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.02,  # Position text slightly above bottom
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    color="white",
                    rotation=90,
                    fontweight="bold",
                    fontsize=12,
                )

        # Place legend outside the plot
        plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc="upper left")

        # Adjust layout to make room for the legend
        plt.tight_layout()

        # Save and show the plot
        if save_fig:
            plt.savefig("bar_plot.png", bbox_inches="tight")
        if show_fig:
            plt.show()


if __name__ == "__main__":
    sample_data = {
        "Logistic Regression": {
            "Kaggle Binary": 0.96,
            "Wikimedia Multiclass": 0.90,
            "Kaggle Multilabel": 0.32,
            "Augmented Multilabel": 0.15,
        },
        "Naive Bayes": {
            "Kaggle Binary": 0.91,
            "Wikimedia Multiclass": 0.80,
            "Kaggle Multilabel": 0.65,
            "Augmented Multilabel": 0.13,
        },
        "Linear SVM": {
            "Kaggle Binary": 0.96,
            "Wikimedia Multiclass": 0.90,
            "Kaggle Multilabel": 0.38,
            "Augmented Multilabel": 0.25,
        },
        "CNN": {
            "Kaggle Binary": 0.50,
            "Wikimedia Multiclass": 0.33,
            "Kaggle Multilabel": 0.20,
            "Augmented Multilabel": 0.20,
        },
        "Transformer": {
            "Kaggle Binary": 0.50,
            "Wikimedia Multiclass": 0.33,
            "Kaggle Multilabel": 0.20,
            "Augmented Multilabel": 0.20,
        },
    }

    visualizer = ModelPerformanceVisualizer()
    visualizer.set_data(sample_data)
    visualizer.bar_plot()
