"""Compare different SVM kernels.

Author: Johannes Krämer
"""

import logging
import sklearn.model_selection as skms
import sklearn.svm as sksvm
import svm.dataset
import svm.evaluation
import svm.features
import svm.tuning

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    random_state = 42
    frac = 0.1

    df = svm.dataset.load_data_frame(frac=frac, random_state=random_state)
    X, y = svm.features.extract_features(df)

    X_train, X_test, y_train, y_test = skms.train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    param_grid = {"C": [0.01, 0.1, 1, 10, 100, 1000]}

    # Linear SVM (LinearSVC)
    svm_model_linearsvc = svm.tuning.tune_model(
        X_train, y_train, sksvm.LinearSVC(), param_grid
    )
    svm.evaluation.evaluate_model(X_test, y_test, svm_model_linearsvc)

    # Linear SVM
    svm_model_linear = svm.tuning.tune_model(
        X_train, y_train, sksvm.SVC(kernel="linear"), param_grid
    )
    svm.evaluation.evaluate_model(X_test, y_test, svm_model_linear)

    # RBF SVM
    svm_model_rbf = svm.tuning.tune_model(
        X_train, y_train, sksvm.SVC(kernel="rbf", gamma="scale"), param_grid
    )
    svm.evaluation.evaluate_model(X_test, y_test, svm_model_rbf)

    # Polynomial SVM
    svm_model_poly = svm.tuning.tune_model(
        X_train, y_train, sksvm.SVC(kernel="poly", gamma="scale"), param_grid
    )
    svm.evaluation.evaluate_model(X_test, y_test, svm_model_poly)

    # Sigmoid SVM
    svm_model_sigmoid = svm.tuning.tune_model(
        X_train, y_train, sksvm.SVC(kernel="sigmoid", gamma="scale"), param_grid
    )
    svm.evaluation.evaluate_model(X_test, y_test, svm_model_sigmoid)


if __name__ == "__main__":
    main()


# 2024-12-08 00:51:28,848 - INFO - LinearSVC() - Tuning
# 2024-12-08 00:51:33,054 - INFO - LinearSVC() - Best params: {'C': 100}
# 2024-12-08 00:51:33,055 - INFO - LinearSVC() - Best score: 0.9547229304889934
# 2024-12-08 00:51:33,055 - INFO - LinearSVC(C=100) - Evaluating model
# 2024-12-08 00:51:33,060 - INFO - LinearSVC(C=100) - Confusion Matrix:
# [[597  24]
#  [ 15 447]]
# 2024-12-08 00:51:33,079 - INFO - LinearSVC(C=100) - Classification Report:
#               precision    recall  f1-score   support
#
#         good       0.98      0.96      0.97       621
#  promotional       0.95      0.97      0.96       462
#
#     accuracy                           0.96      1083
#    macro avg       0.96      0.96      0.96      1083
# weighted avg       0.96      0.96      0.96      1083
#
# 2024-12-08 00:51:33,081 - INFO - LinearSVC(C=100) - Accuracy: 0.96398891966759

# 2024-12-08 00:51:33,081 - INFO - SVC(kernel='linear') - Tuning
# 2024-12-08 00:53:52,728 - INFO - SVC(kernel='linear') - Best params: {'C': 10}
# 2024-12-08 00:53:52,729 - INFO - SVC(kernel='linear') - Best score: 0.9547229304889934
# 2024-12-08 00:53:52,729 - INFO - SVC(C=10, kernel='linear') - Evaluating model
# 2024-12-08 00:53:58,714 - INFO - SVC(C=10, kernel='linear') - Confusion Matrix:
# [[597  24]
#  [ 15 447]]
# 2024-12-08 00:53:58,731 - INFO - SVC(C=10, kernel='linear') - Classification Report:
#               precision    recall  f1-score   support
#
#         good       0.98      0.96      0.97       621
#  promotional       0.95      0.97      0.96       462
#
#     accuracy                           0.96      1083
#    macro avg       0.96      0.96      0.96      1083
# weighted avg       0.96      0.96      0.96      1083
#
# 2024-12-08 00:53:58,734 - INFO - SVC(C=10, kernel='linear') - Accuracy: 0.96398891966759

# 2024-12-08 00:53:58,734 - INFO - SVC() - Tuning
# 2024-12-08 00:56:59,433 - INFO - SVC() - Best params: {'C': 10}
# 2024-12-08 00:56:59,434 - INFO - SVC() - Best score: 0.9496420990802174
# 2024-12-08 00:56:59,435 - INFO - SVC(C=10) - Evaluating model
# 2024-12-08 00:57:08,818 - INFO - SVC(C=10) - Confusion Matrix:
# [[597  24]
#  [ 19 443]]
# 2024-12-08 00:57:08,836 - INFO - SVC(C=10) - Classification Report:
#               precision    recall  f1-score   support
#
#         good       0.97      0.96      0.97       621
#  promotional       0.95      0.96      0.95       462
#
#     accuracy                           0.96      1083
#    macro avg       0.96      0.96      0.96      1083
# weighted avg       0.96      0.96      0.96      1083
#
# 2024-12-08 00:57:08,839 - INFO - SVC(C=10) - Accuracy: 0.9602954755309326

# 2024-12-08 00:57:08,839 - INFO - SVC(kernel='poly') - Tuning
# 2024-12-08 01:00:35,033 - INFO - SVC(kernel='poly') - Best params: {'C': 10}
# 2024-12-08 01:00:35,033 - INFO - SVC(kernel='poly') - Best score: 0.9212289578021332
# 2024-12-08 01:00:35,034 - INFO - SVC(C=10, kernel='poly') - Evaluating model
# 2024-12-08 01:00:47,868 - INFO - SVC(C=10, kernel='poly') - Confusion Matrix:
# [[577  44]
#  [ 38 424]]
# 2024-12-08 01:00:47,884 - INFO - SVC(C=10, kernel='poly') - Classification Report:
#               precision    recall  f1-score   support
#
#         good       0.94      0.93      0.93       621
#  promotional       0.91      0.92      0.91       462
#
#     accuracy                           0.92      1083
#    macro avg       0.92      0.92      0.92      1083
# weighted avg       0.92      0.92      0.92      1083
#
# 2024-12-08 01:00:47,887 - INFO - SVC(C=10, kernel='poly') - Accuracy: 0.9242843951985226

# 2024-12-08 01:00:47,888 - INFO - SVC(kernel='sigmoid') - Tuning
# 2024-12-08 01:02:32,612 - INFO - SVC(kernel='sigmoid') - Best params: {'C': 10}
# 2024-12-08 01:02:32,613 - INFO - SVC(kernel='sigmoid') - Best score: 0.9549538773712104
# 2024-12-08 01:02:32,614 - INFO - SVC(C=10, kernel='sigmoid') - Evaluating model
# 2024-12-08 01:02:37,560 - INFO - SVC(C=10, kernel='sigmoid') - Confusion Matrix:
# [[595  26]
#  [ 16 446]]
# 2024-12-08 01:02:37,577 - INFO - SVC(C=10, kernel='sigmoid') - Classification Report:
#               precision    recall  f1-score   support
#
#         good       0.97      0.96      0.97       621
#  promotional       0.94      0.97      0.96       462
#
#     accuracy                           0.96      1083
#    macro avg       0.96      0.96      0.96      1083
# weighted avg       0.96      0.96      0.96      1083
#
# 2024-12-08 01:02:37,579 - INFO - SVC(C=10, kernel='sigmoid') - Accuracy: 0.961218836565097
