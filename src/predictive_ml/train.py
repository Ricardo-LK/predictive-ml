import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    RepeatedStratifiedKFold,
    cross_validate,
    train_test_split,
    cross_val_predict
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    make_scorer,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predictive_ml.data import load_data

RANDOM_STATE = 2

def build_preprocessor(Xtr):
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

    return preprocessing


def eval_models(Xtr, ytr, preprocessing, model_cv, models):

    scoring = {
        "precision": make_scorer( precision_score, zero_division=0 ),
        "recall": make_scorer( recall_score, zero_division=0 ),
        "f1": make_scorer( f1_score, zero_division=0 ),
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    results = []
    for model_name, classifier in models.items():
        print(f"\nEvaluating: {model_name}")

        pipeline = make_pipeline( preprocessing, classifier )

        scores = cross_validate( pipeline, Xtr, ytr, cv=model_cv, scoring=scoring, n_jobs=-1)

        result = {
            "model": model_name,
            "precision": scores["test_precision"].mean(),
            "recall": scores["test_recall"].mean(),
            "f1": scores["test_f1"].mean(),
            "roc_auc": scores["test_roc_auc"].mean(),
            "pr_auc": scores["test_pr_auc"].mean(),
        }

        results.append(result)

    result_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False,)
    return result_df


def eval_thresholds(model, Xtr, ytr, threshold_cv):

    probabilities = cross_val_predict(
        model,
        Xtr, ytr,
        cv = threshold_cv, 
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    thresholds = np.arange(0.1, 0.91, 0.05)

    results = []
    for threshold in thresholds:
        predict = (probabilities >= threshold).astype(int)

        result = {
            "threshold": threshold,
            "precision": precision_score(ytr, predict),
            "recall": recall_score(ytr, predict),
            "f1": f1_score(ytr, predict),
        }

        results.append(result)

    result_df = pd.DataFrame(results).sort_values("f1", ascending=False,)

    return result_df


def eval_final(model, Xtr, ytr, Xte, yte, threshold):

    # Training for test with best parameters
    model.fit(Xtr, ytr)

    probabilities = model.predict_proba(Xte)[:, 1]
    predict = (probabilities >= threshold).astype(int)

    results = {
        "precision": precision_score(yte, predict),
        "recall": recall_score(yte, predict),
        "f1": f1_score(yte, predict),
        "roc_auc": roc_auc_score(yte, probabilities),
        "pr_auc": average_precision_score(yte, probabilities),
    }

    return results, confusion_matrix(yte, predict)


def save_model(model, threshold):

    Path("models").mkdir(exist_ok=True)

    artifact = {
        "model": model,
        "threshold": threshold,
    }

    joblib.dump(
        artifact,
        "models/predictive_model.joblib"
    )


# Main
def main():
    X, y = load_data()

    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessing = build_preprocessor(Xtr)
    model_cv = RepeatedStratifiedKFold( n_splits=5, n_repeats=5, random_state=RANDOM_STATE ) # Using repeated cross val to reduce dependency on a single data split

    # Choosing best model
    models = {
        "dummy": DummyClassifier(strategy = "prior" ),
        "logistic_regression": LogisticRegression( max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE ),
        "random_forest": RandomForestClassifier( n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    }
    results_df = eval_models(Xtr, ytr, preprocessing, model_cv, models)

    print("\nModel comparison:")
    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    best_model_name = results_df.iloc[0]["model"]
    print(f"\nBest model: {best_model_name}")

    # Choosing best threshold
    threshold_cv = StratifiedKFold( n_splits=5, shuffle=True, random_state=RANDOM_STATE ) # cross_val_predict needs each sample to be validated only once
    best_classifier = models[best_model_name]
    model = make_pipeline( preprocessing, best_classifier )
    threshold_results = eval_thresholds(model, Xtr, ytr, threshold_cv)

    print("\nThreshold comparison:")
    print(
        threshold_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    best_threshold = threshold_results.iloc[0]["threshold"]
    print(f"\nBest threshold: {best_threshold:.2f}")

    # Final evaluation
    final_results, matrix = eval_final(model, Xtr, ytr, Xte, yte, best_threshold)
    for metric, score in final_results.items():
        print(f"{metric}: {score:.4f}")

    print("\nConfusion matrix:")
    print(matrix)

    # Storing best model
    save_model( model, best_threshold )
    print("\nModel saved.")


if __name__ == "__main__":
    main()