from ucimlrepo import fetch_ucirepo

from predictive_ml.artifact import FEATURE_COLUMNS

UCI_DATASET_ID = 601  # AI4I 2020 Predictive Maintenance Dataset
TARGET_COLUMN = "Machine failure"


def load_data():
    try:
        dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    except Exception as exc:  # ucimlrepo raises several error types
        raise RuntimeError(
            f"Could not download UCI dataset id={UCI_DATASET_ID} (AI4I 2020). "
            "Training needs internet access to archive.ics.uci.edu. "
            f"Original error: {exc}"
        ) from exc

    features = dataset.data.features
    missing = [c for c in FEATURE_COLUMNS if c not in features.columns]
    if missing:
        raise RuntimeError(
            f"UCI dataset schema changed: missing feature columns {missing}; "
            f"got {list(features.columns)}"
        )

    # Explicit selection keeps training aligned with the API feature contract.
    X = features[FEATURE_COLUMNS].copy()

    targets = dataset.data.targets.copy()
    y = targets[TARGET_COLUMN].astype(int)

    return X, y

if __name__ == '__main__':
    X, y = load_data()

    print("Features:")
    print(X.head())
    print("\nFeatures shape: ", X.shape)


    print("\nTarget distribution:")
    print(y.value_counts())
    print(y.value_counts(normalize=True))
