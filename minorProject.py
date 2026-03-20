# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:10:59 2026

@author: ash31
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

def load_data(filepath):
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df = df.sort_values("date")
    return df

def create_result(df):
    df["result"] = np.where(df["goal_home_ft"] > df["goal_away_ft"], "H", np.where(df["goal_home_ft"] < df["goal_away_ft"], "A", "D"))
    return df

def create_features(df):
    df["performance_diff"] = df["performance_acum_home"] - df["performance_acum_away"]
    
    return df
    
def select_features(df):
    features = ["performance_diff"]    
    df = df.dropna(subset=features)
    X = df[features]
    y = df["result"]
    return X, y, features

def split_data(df, features):
    print("Available seasons:")
    print(df["season"].unique())
    train = df[df["season"] == "18/19"]
    test = df[df["season"] == "19/20"]     
    X_train = train[features]
    y_train = train["result"]
    X_test = test[features]
    y_test = test["result"]
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test     

def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
        }       
    model_names = []
    accuracies = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(name, "Accuracy:", acc)
        model_names.append(name)
        accuracies.append(acc)
    return model_names, accuracies

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
    

main()