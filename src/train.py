import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import joblib
import os
from data_processing import process_data

def train_and_evaluate():
    # Load processed data
    df = process_data()
    X = df.drop(['CustomerId', 'is_high_risk'], axis=1)
    y = df['is_high_risk']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define models and hyperparameters
    models = {
        'LogisticRegression': LogisticRegression(solver='liblinear', random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }
    param_grids = {
        'LogisticRegression': {
            'C': [0.01, 0.1, 1, 10]
        },
        'GradientBoosting': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    }

    best_score = 0
    best_model = None
    best_model_name = ''
    best_metrics = {}

    mlflow.set_experiment('credit-risk-model')
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            grid = GridSearchCV(model, param_grids[name], cv=3, scoring='f1', n_jobs=-1)
            grid.fit(X_train, y_train)
            y_pred = grid.predict(X_test)
            y_proba = grid.predict_proba(X_test)[:, 1] if hasattr(grid, 'predict_proba') else None

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
            }
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(grid.best_estimator_, f'{name}_model')

            if metrics['f1'] > best_score:
                best_score = metrics['f1']
                best_model = grid.best_estimator_
                best_model_name = name
                best_metrics = metrics

    # Save best model
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    print(f'Best model: {best_model_name}')
    print('Metrics:', best_metrics)

if __name__ == '__main__':
    train_and_evaluate()
