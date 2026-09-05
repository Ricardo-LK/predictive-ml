from ucimlrepo import fetch_ucirepo

def load_data():
    dataset = fetch_ucirepo(id=601)

    X = dataset.data.features.copy()
    
    targets = dataset.data.targets.copy()
    y = targets["Machine failure"].astype(int)

    return X, y

if __name__ == '__main__':
    X, y = load_data()
    
    print("Features:")
    print(X.head())
    print("\nFeatures shape: ", X.shape)


    print("\nTarget distribution:")
    print(y.value_counts())
    print(y.value_counts(normalize=True))