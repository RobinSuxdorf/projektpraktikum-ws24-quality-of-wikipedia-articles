import logging
import sklearn.model_selection as skms
import sklearn.svm as sksvm
import svm.dataset
import svm.evalutation
import svm.features
import svm.tuning

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    random_state = 42
    frac = 1.0

    df = svm.dataset.load_data_frame(frac=frac, random_state=random_state)
    X, y = svm.features.extract_features(df)

    X_train, X_test, y_train, y_test = skms.train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    svm_model_linearsvc = sksvm.LinearSVC(C=1)
    svm_model_linearsvc.fit(X_train, y_train)
    svm.evalutation.evaluate_model(X_test, y_test, svm_model_linearsvc)


if __name__ == "__main__":
    main()
