import logging
import sklearn.model_selection as skms
import sklearn.multioutput as skmo
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
    frac = 0.2

    df = svm.dataset.load_data_frame(frac=frac, random_state=random_state)
    X, y = svm.features.extract_features_promotional_categories(df)

    X_train, X_test, y_train, y_test = skms.train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Throws ValueError: WRITEBACKIFCOPY base is read-only when using n_jobs=-1
    svm_model = skmo.MultiOutputClassifier(
        sksvm.SVC(kernel="sigmoid", gamma="scale"), n_jobs=1
    )
    svm_model.fit(X_train, y_train)
    svm.evaluation.evaluate_model_with_categories(X_test, y_test, svm_model)


if __name__ == "__main__":
    main()
