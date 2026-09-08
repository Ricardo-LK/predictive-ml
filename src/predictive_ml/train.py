import pandas as pd
import numpy as np
import joblib
from pathlib import Path

import mlflow
import mlflow.sklearn
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

EXPERIMENT_NAME = "predictive-maintenance"
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
        "average_precision": "average_precision",
    }

    results = []
    for model_name, classifier in models.items():
        print(f"\nEvaluating: {model_name}")

        pipeline = make_pipeline( preprocessing, classifier )

        with mlflow.start_run(run_name=f"cv_{model_name}"):
            scores = cross_validate( pipeline, Xtr, ytr, cv=model_cv, scoring=scoring, n_jobs=-1)

            result = {
                "model": model_name,
                "precision": scores["test_precision"].mean(),
                "recall": scores["test_recall"].mean(),
                "f1": scores["test_f1"].mean(),
                "roc_auc": scores["test_roc_auc"].mean(),
                "average_precision": scores[
                    "test_average_precision"
                ].mean(),
            }

            results.append(result)

            # MLflow
            params = {
                "model": model_name,
                "random_state": RANDOM_STATE,
                "cv_folds": model_cv.cvargs["n_splits"],
                "cv_repeats": model_cv.n_repeats,
            }
            for key, value in classifier.get_params().items():
                params[f"model_{key}"] = value

            metrics = {
                "cv_precision_mean": result["precision"],
                "cv_recall_mean": result["recall"],
                "cv_f1_mean": result["f1"],
                "cv_roc_auc_mean": result["roc_auc"],
                "cv_average_precision_mean": result["average_precision"],

                "cv_precision_std": scores["test_precision"].std(),
                "cv_recall_std": scores["test_recall"].std(),
                "cv_f1_std": scores["test_f1"].std(),
                "cv_roc_auc_std": scores["test_roc_auc"].std(),
                "cv_average_precision_std": scores[
                    "test_average_precision"
                ].std(),
            }
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

    result_df = pd.DataFrame(results).sort_values("average_precision", ascending=False,)
    return result_df


def eval_thresholds(model, Xtr, ytr, threshold_cv):

    probabilities = cross_val_predict(
        model,
        Xtr, ytr,
        cv = threshold_cv, 
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    thresholds = np.round(np.arange(0.1, 0.91, 0.05))

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
        "average_precision": average_precision_score(yte, probabilities),
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
    mlflow.set_experiment(EXPERIMENT_NAME)
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
    # MLflow for final model
    final_params = {
        "model": best_model_name,
        "threshold": best_threshold,
        "random_state": RANDOM_STATE
    }
    with mlflow.start_run(run_name=f"final_{best_model_name}"):
        mlflow.log_params(final_params)
        for metric, score in final_results.items():
            mlflow.log_metric( f"test_{metric}", score )

        mlflow.sklearn.log_model(sk_model=model, name="model", skops_trusted_types=["numpy.dtype"])

    for metric, score in final_results.items():
        print(f"{metric}: {score:.4f}")

    print("\nConfusion matrix:")
    print(matrix)

    # Storing best model
    save_model( model, best_threshold )
    print("\nModel saved.")


if __name__ == "__main__":
    main()