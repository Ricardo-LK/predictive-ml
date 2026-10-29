from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_models(random_state):

    models = {
        "dummy": DummyClassifier(
            strategy="prior"
        ),

        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
    }

    return models