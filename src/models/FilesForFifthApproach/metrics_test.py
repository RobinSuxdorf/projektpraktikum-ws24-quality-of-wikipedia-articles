from sklearn import metrics


def calculate_metrics(predicted_values, real_values):
    print("Accuracy: " + str(metrics.accuracy_score(predicted_values, real_values)))
    print("Recall: " + str(metrics.recall_score(predicted_values, real_values, average="micro")))
