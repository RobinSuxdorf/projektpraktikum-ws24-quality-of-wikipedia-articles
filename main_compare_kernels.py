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
