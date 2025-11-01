import logging

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


def main():
    """
    Performs the training and evaluation of baseline models using sklearn Pipelines.
    """
    logging.info("--- Starting Baseline Model Training ---")

    # Load data
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    
    # --- 1. Prepare Data and Feature Sets ---
    # Convert target to binary format
    train_df[TARGET_CLASSIFICATION] = train_df[TARGET_CLASSIFICATION].map({'Yes': 1, 'No': 0})
    val_df[TARGET_CLASSIFICATION] = val_df[TARGET_CLASSIFICATION].map({'Yes': 1, 'No': 0})

    # Define feature sets from config
    all_numeric_features = train_df.select_dtypes(include=['number']).columns.drop(
        [TARGET_CLASSIFICATION, TARGET_REGRESSION] + DROP_FEATURES, errors='ignore'
    ).tolist()
    
    numeric_features_clf = [col for col in all_numeric_features if col not in CLASSIFICATION_DROP_COLS]
    numeric_features_reg = [col for col in all_numeric_features if col not in REGRESSION_DROP_COLS]

    # --- 2. Separate Features (X) and Targets (y) ---
    # Master feature sets
    X_train = train_df.drop(columns=DROP_FEATURES + [TARGET_CLASSIFICATION, TARGET_REGRESSION])
    X_val = val_df.drop(columns=DROP_FEATURES + [TARGET_CLASSIFICATION, TARGET_REGRESSION])

    # Task-specific targets
    y_train_clf = train_df[TARGET_CLASSIFICATION]
    y_val_clf = val_df[TARGET_CLASSIFICATION]
    y_train_reg = train_df[TARGET_REGRESSION]
    y_val_reg = val_df[TARGET_REGRESSION]

    # Task-specific feature sets by selecting only the columns we need
    X_train_clf = X_train[numeric_features_clf + CATEGORICAL_FEATURES]
    X_val_clf = X_val[numeric_features_clf + CATEGORICAL_FEATURES]

    X_train_reg = X_train[numeric_features_reg + CATEGORICAL_FEATURES]
    X_val_reg = X_val[numeric_features_reg + CATEGORICAL_FEATURES]

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

    # --- 4. Train and Log Classification Models ---
    logging.info("Training classification models...")
    for model_name, model in classification_models.items():
        with mlflow.start_run(run_name=f"{model_name}_clf_pipeline"):
            # Create the full pipeline: preprocessor -> model
            preprocessor = create_preprocessing_pipeline(numeric_features_clf, CATEGORICAL_FEATURES)
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])

            # Fit the pipeline on the training data
            pipeline.fit(X_train_clf, y_train_clf)
            
            # Evaluate on the validation set
            y_pred_val = pipeline.predict(X_val_clf)
            accuracy = accuracy_score(y_val_clf, y_pred_val)
            f1 = f1_score(y_val_clf, y_pred_val, average='weighted')

            # Log parameters, metrics, and the entire pipeline
            mlflow.log_param("model_type", "classification")
            mlflow.log_params(model.get_params())
            mlflow.log_metric("val_accuracy", accuracy)
            mlflow.log_metric("val_f1_score", f1)
            mlflow.sklearn.log_model(pipeline, model_name)
            
            logging.info(f"  {model_name} -> Val F1: {f1:.4f}, Val Accuracy: {accuracy:.4f}")

    # --- 5. Train and Log Regression Models ---
    logging.info("Training regression models...")
    for model_name, model in regression_models.items():
        with mlflow.start_run(run_name=f"{model_name}_reg_pipeline"):
            # Create the full pipeline
            preprocessor = create_preprocessing_pipeline(numeric_features_reg, CATEGORICAL_FEATURES)
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])

            # Fit and evaluate
            pipeline.fit(X_train_reg, y_train_reg)
            y_pred_val = pipeline.predict(X_val_reg)
            mae = mean_absolute_error(y_val_reg, y_pred_val)
            rmse = root_mean_squared_error(y_val_reg, y_pred_val)

            # Log everything
            mlflow.log_param("model_type", "regression")
            mlflow.log_params(model.get_params())
            mlflow.log_metric("val_mae", mae)
            mlflow.log_metric("val_rmse", rmse)
            mlflow.sklearn.log_model(pipeline, model_name)

            logging.info(f"  {model_name} -> Val MAE: {mae:.4f}, Val RMSE: {rmse:.4f}")

    logging.info("--- Model Training Complete ---")

if __name__ == "__main__":
    main()