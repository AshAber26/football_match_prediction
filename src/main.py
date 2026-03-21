def main():
    filepath = "C:/Users/ash31/OneDrive - Aberystwyth University/Documents/preprocessed.csv"
    df = load_data(filepath)
    df = create_result(df)
    df = create_features(df)
    df = df.dropna()
    X, y, features = select_features(df)
    X_train, X_test, y_train, y_test = split_data(df, features)
    X_train, X_test = scale_features(X_train, X_test)
    model_names, accuracies = train_models(X_train, X_test, y_train, y_test)
    
