from pathlib import Path

import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predictive_ml.data import load_data

RANDOM_STATE = 2

def build_model(Xtr):
    num_features = Xtr.select_dtypes(include="number").columns.tolist()
    categ_features = Xtr.select_dtypes(exclude="number").columns.tolist()

    print("Numerical features:")
    print(num_features)

    print("\nCategorical features:")
    print(categ_features)

    # Pipeline
    num_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )

    categ_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )

    # Preprocessing
    preprocessing = ColumnTransformer(
        transformers = [ ("num", num_pipeline, num_features), ("categ", categ_pipeline, categ_features) ]
    )

    classifier = LogisticRegression( max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)

    model = make_pipeline( preprocessing, classifier )

    return model


def main():
    X, y = load_data()

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTrain shape:", Xtr.shape)
    print("Test shape:", Xte.shape)

    model = build_model(Xtr)

    # Train
    model.fit(Xtr, ytr)

    # Test
    predictions = model.predict(Xte)
    probabilities = model.predict_proba(Xte)[:, 1]

    print("\nClassification report:")
    print(
        classification_report(
            yte,
            predictions,
            digits=4,
        )
    )

    print("\nConfusion matrix:")
    print(confusion_matrix(yte, predictions))

    print(
        "\nROC-AUC:",
        roc_auc_score(yte, probabilities),
    )

    print(
        "PR-AUC:",
        average_precision_score(
            yte,
            probabilities,
        ),
    )

    Path("models").mkdir(exist_ok=True)

    joblib.dump(
        model,
        "models/baseline_model.joblib",
    )

    print("\nModel saved.")


if __name__ == "__main__":
    main()