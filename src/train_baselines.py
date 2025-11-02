import logging
from typing import List, Any

import pandas as pd
import mlflow

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from src.features import create_preprocessing_pipeline

from src.config import (
    TRAIN_PATH, VAL_PATH, TARGET_CLASSIFICATION, TARGET_REGRESSION,
    MLFLOW_EXPERIMENT_NAME, RANDOM_STATE, CATEGORICAL_FEATURES, DROP_FEATURES,
    CLASSIFICATION_DROP_COLS, REGRESSION_DROP_COLS
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _train_and_log_model(
    model: Any,
    model_name: str,
    task_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    numeric_features: List[str],
    categorical_features: List[str]
) -> None:
    """Creates a pipeline, trains a model, and logs results to MLflow.

    Args:
        model: The scikit-learn model instance to train.
        model_name: The name of the model for logging.
        task_type: 'classification' or 'regression'.
        X_train, y_train: Training data and labels.
        X_val, y_val: Validation data and labels.
        numeric_features: List of numeric feature names.
        categorical_features: List of categorical feature names.
    """
    run_name = f"{model_name}_{task_type.split('c')[0]}_pipeline" # Generates 'clf' or 'reg'
    with mlflow.start_run(run_name=run_name):
        # Create and fit the full pipeline
        preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), (task_type, model)])
        pipeline.fit(X_train, y_train)

        # Evaluate on the validation set
        y_pred_val = pipeline.predict(X_val)

        # Log parameters, metrics, and the pipeline artifact
        mlflow.log_param("model_type", task_type)
        mlflow.log_params(model.get_params())
        
        if task_type == "classification":
            accuracy = accuracy_score(y_val, y_pred_val)
            f1 = f1_score(y_val, y_pred_val, average='weighted')
            mlflow.log_metric("val_accuracy", accuracy)
            mlflow.log_metric("val_f1_score", f1)
            logging.info(f"  {model_name} -> Val F1: {f1:.4f}, Val Accuracy: {accuracy:.4f}")
        else: # Regression
            mae = mean_absolute_error(y_val, y_pred_val)
            rmse = root_mean_squared_error(y_val, y_pred_val)
            mlflow.log_metric("val_mae", mae)
            mlflow.log_metric("val_rmse", rmse)
            logging.info(f"  {model_name} -> Val MAE: {mae:.4f}, Val RMSE: {rmse:.4f}")

        mlflow.sklearn.log_model(pipeline, model_name)

def main():
    """Orchestrates the training of all baseline models."""
    # --- 1. Load Data ---
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    
    # --- 2. Prepare Data and Feature Sets ---
    train_df[TARGET_CLASSIFICATION] = train_df[TARGET_CLASSIFICATION].map({'Yes': 1, 'No': 0})
    val_df[TARGET_CLASSIFICATION] = val_df[TARGET_CLASSIFICATION].map({'Yes': 1, 'No': 0})

    all_numeric_features = train_df.select_dtypes(include=['number']).columns.drop(
        [TARGET_CLASSIFICATION, TARGET_REGRESSION] + DROP_FEATURES, errors='ignore'
    ).tolist()
    
    numeric_features_clf = [col for col in all_numeric_features if col not in CLASSIFICATION_DROP_COLS]
    numeric_features_reg = [col for col in all_numeric_features if col not in REGRESSION_DROP_COLS]

    X_train = train_df.drop(columns=DROP_FEATURES + [TARGET_CLASSIFICATION, TARGET_REGRESSION])
    X_val = val_df.drop(columns=DROP_FEATURES + [TARGET_CLASSIFICATION, TARGET_REGRESSION])
    
    # --- 3. Define Models ---
    classification_models = {
        "LogisticRegression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=RANDOM_STATE)
    }
    regression_models = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=RANDOM_STATE)
    }

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # --- 4. Train All Models using the Helper Function ---
    logging.info("Training classification models...")
    for name, model in classification_models.items():
        _train_and_log_model(
            model=model, model_name=name, task_type="classification",
            X_train=X_train[numeric_features_clf + CATEGORICAL_FEATURES], y_train=train_df[TARGET_CLASSIFICATION],
            X_val=X_val[numeric_features_clf + CATEGORICAL_FEATURES], y_val=val_df[TARGET_CLASSIFICATION],
            numeric_features=numeric_features_clf, categorical_features=CATEGORICAL_FEATURES
        )

    logging.info("Training regression models...")
    for name, model in regression_models.items():
        _train_and_log_model(
            model=model, model_name=name, task_type="regression",
            X_train=X_train[numeric_features_reg + CATEGORICAL_FEATURES], y_train=train_df[TARGET_REGRESSION],
            X_val=X_val[numeric_features_reg + CATEGORICAL_FEATURES], y_val=val_df[TARGET_REGRESSION],
            numeric_features=numeric_features_reg, categorical_features=CATEGORICAL_FEATURES
        )

    logging.info("--- Model Training Complete ---")

if __name__ == "__main__":
    main()