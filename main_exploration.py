import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import svm.dataset
import svm.evaluation
import svm.features
import svm.tuning

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha="center")


def main():
    random_state = 42
    frac = 1.0

    df = svm.dataset.load_data_frame(frac=frac, random_state=random_state)

    logging.info("Dataset shape: %s", df.shape)
    logging.info("Dataset info:\n%s", df.info())

    logging.info("First few rows of the dataset:\n%s", df.head())

    label_counts = df["label"].value_counts()
    logging.info("Label distribution:\n%s", label_counts)

    promo_labels = ["advert", "coi", "fanpov", "pr", "resume"]
    for label in promo_labels:
        promo_label_counts = df[label].value_counts()
        logging.info("Distribution of '%s' label:\n%s", label, promo_label_counts)

    directory = "distribution"

    if not os.path.exists(directory):
        os.makedirs(directory)

    logging.info("Plotting distribution")
    plt.figure(figsize=(10, 6))
    sns.countplot(x="label", data=df)
    plt.title("Distribution")
    addlabels(df["label"].unique(), label_counts.values)
    plt.savefig("distribution/distribution.png")
    plt.close()

    df_promo = df[df["label"] == "promotional"].copy()

    for label in promo_labels:
        logging.info("Plotting distribution of %s", label)
        plt.figure(figsize=(10, 6))
        sns.countplot(x=label, data=df_promo)
        plt.title(f"Distribution of '{label}'")
        addlabels(df_promo[label].unique(), df_promo[label].value_counts().values)
        plt.savefig(f"distribution/distribution_promo_{label}.png")
        plt.close()

    df_promo.loc[:, "label_combination"] = df_promo[promo_labels].apply(
        lambda row: "_".join(row.index[row == 1]), axis=1
    )
    label_combination_counts = df_promo["label_combination"].value_counts()
    plt.figure(figsize=(16, 10))
    sns.barplot(x=label_combination_counts.index, y=label_combination_counts.values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Label Combinations in Promotional Articles")
    plt.xlabel("Label Combination")
    plt.ylabel("Count")
    addlabels(label_combination_counts.index, label_combination_counts.values)
    plt.tight_layout()
    plt.savefig("distribution/label_combinations.png")
    plt.close()


if __name__ == "__main__":
    main()
