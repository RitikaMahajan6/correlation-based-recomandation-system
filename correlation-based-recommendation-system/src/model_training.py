# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

def train_model(processed_data):
    # Assuming processed_data is a DataFrame with features and a target variable
    X = processed_data.drop('target', axis=1)  # Replace 'target' with the actual target column name
    y = processed_data['target']  # Replace 'target' with the actual target column name

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    model = RandomForestClassifier()

    # Start MLflow run
    with mlflow.start_run():
        # Train the model
        model.fit(X_train, y_train)

        # Log parameters and metrics
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_metric("accuracy", model.score(X_test, y_test))

        # Log the model
        mlflow.sklearn.log_model(model, "model")

    return model