import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class ModelPerformanceVisualizer:
    def __init__(self):
        """Initialize the visualizer with default style settings."""
        sns.set_theme(style="whitegrid")
        self.colors = sns.color_palette("muted", 5)
        self.fig_size = (12, 8)

    def set_data(self, data: dict, kpi: str):
        """
        Set the performance data for visualization.

        Args:
            data (dict): A dictionary containing model performance data.
            kpi (str): The key performance indicator to visualize.
        """
        self.kpi = kpi
        rows = []
        for model, usecases in data.items():
            for usecase, value in usecases.items():
                rows.append({"model": model, "usecase": usecase, self.kpi: value})
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
            y=self.kpi,
            hue="model",
            palette=self.colors,
            ax=ax,
        )

        # Customize the plot
        ax.set_title(
            f"Model Performance Comparison (Macro Avg {self.kpi})", fontsize=16
        )
        ax.set_xlabel("Use Case", fontsize=14)
        ax.set_ylabel(f"Macro Average {self.kpi}", fontsize=14)
        ax.set_ylim(0, 1.0)

        # Add value labels at the bottom of bars with white text
        for i, container in enumerate(bars.containers):
            for j, bar in enumerate(container):
                value = self.data[
                    (self.data["model"] == self.data["model"].unique()[i])
                    & (self.data["usecase"] == self.data["usecase"].unique()[j])
                ][self.kpi].values[0]
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
    # Macro Average Precision
    precision_data = {
        "Logistic Regression": {
            "Kaggle Binary": 0.96,
            "Wikimedia Multiclass": 0.90,
            "Kaggle Multilabel": 0.44,
            # "Augmented Multilabel": 0.30,
        },
        "Multinominal Naive Bayes": {
            "Kaggle Binary": 0.91,
            "Wikimedia Multiclass": 0.80,
            "Kaggle Multilabel": 0.34,
            # "Augmented Multilabel": 0.27,
        },
        "Linear SVM": {
            "Kaggle Binary": 0.97,
            "Wikimedia Multiclass": 0.90,
            "Kaggle Multilabel": 0.38,
            # "Augmented Multilabel": 0.23,
        },
        "Künstliches Neuronales Netz": {
            "Kaggle Binary": 0.96,
            "Wikimedia Multiclass": 0.89,
            "Kaggle Multilabel": 0.43,
            # "Augmented Multilabel": 0.16,
        },
        "Transformer": {
            "Kaggle Binary": 0.95,
            "Wikimedia Multiclass": 0.86,
            "Kaggle Multilabel": 0.38,
            # "Augmented Multilabel": 0.42,
        },
    }
    # Macro Average Recall
    recall_data = {
        "Logistic Regression": {
            "Kaggle Binary": 0.96,
            "Wikimedia Multiclass": 0.90,
            "Kaggle Multilabel": 0.32,
            # "Augmented Multilabel": 0.15,
        },
        "Multinominal Naive Bayes": {
            "Kaggle Binary": 0.91,
            "Wikimedia Multiclass": 0.80,
            "Kaggle Multilabel": 0.65,
            # "Augmented Multilabel": 0.13,
        },
        "Linear SVM": {
            "Kaggle Binary": 0.96,
            "Wikimedia Multiclass": 0.90,
            "Kaggle Multilabel": 0.38,
            # "Augmented Multilabel": 0.25,
        },
        "Künstliches Neuronales Netz": {
            "Kaggle Binary": 0.96,
            "Wikimedia Multiclass": 0.89,
            "Kaggle Multilabel": 0.35,
            # "Augmented Multilabel": 0.20,
        },
        "Transformer": {
            "Kaggle Binary": 0.95,
            "Wikimedia Multiclass": 0.86,
            "Kaggle Multilabel": 0.46,
            # "Augmented Multilabel": 0.50,
        },
    }

    visualizer = ModelPerformanceVisualizer()
    visualizer.set_data(precision_data, "Precision")
    # visualizer.set_data(recall_data, "Recall")
    visualizer.bar_plot()
#
