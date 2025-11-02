import logging

import pandas as pd
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, mean_absolute_error, root_mean_squared_error

from src.config import (
    TEST_PATH, MLFLOW_EXPERIMENT_NAME, TARGET_CLASSIFICATION, TARGET_REGRESSION,
    PLOTS_DIR, TABLES_DIR, DROP_FEATURES
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _evaluate_best_pipeline(
    task_type: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    filter_string: str,
    order_by: str
) -> None:
    """Finds the best pipeline in MLflow for a task, evaluates it, and saves artifacts.
    
    Args:
        task_type: 'classification' or 'regression'.
        X_test: Test feature DataFrame.
        y_test: Test target Series.
        filter_string: MLflow filter string to find relevant runs.
        order_by: MLflow order_by string to sort runs.
    Returns:
        None
    """
    logging.info(f"Evaluating best {task_type} pipeline...")
    
    # Find the best run based on the specified validation metric
    best_runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT_NAME],
        filter_string=filter_string,
        order_by=[order_by],
        max_results=1
    )

    if best_runs.empty:
        logging.warning(f"No {task_type} models found. Skipping.")
        return

    best_run_id = best_runs.iloc[0]["run_id"]
    model_name = best_runs.iloc[0]['tags.mlflow.runName'].split('_')[0]
    model_uri = f"runs:/{best_run_id}/{model_name}"
    
    logging.info(f"Loading best {task_type} pipeline from run: {best_run_id}")
    best_pipeline = mlflow.sklearn.load_model(model_uri)
    
    y_pred_test = best_pipeline.predict(X_test)
    
    if task_type == "classification":
        accuracy = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test, average='weighted')
        logging.info(f"  Test Accuracy: {accuracy:.4f}, Test F1-Score: {f1:.4f}")
        
        metrics_df = pd.DataFrame({"Metric": ["Accuracy", "F1-Score"], "Value": [accuracy, f1]})
        metrics_df.to_csv(TABLES_DIR / "classification_test_metrics.csv", index=False)
        
        cm = confusion_matrix(y_test, y_pred_test)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        plt.title('Confusion Matrix on Test Set'); plt.xlabel('Predicted'); plt.ylabel('Actual')
        plt.savefig(PLOTS_DIR / "confusion_matrix.png"); plt.close()
    
    elif task_type == "regression":
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = root_mean_squared_error(y_test, y_pred_test)
        logging.info(f"  Test MAE: {mae:.4f}, Test RMSE: {rmse:.4f}")
        
        metrics_df = pd.DataFrame({"Metric": ["MAE", "RMSE"], "Value": [mae, rmse]})
        metrics_df.to_csv(TABLES_DIR / "regression_test_metrics.csv", index=False)
        
        residuals = y_test - y_pred_test
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred_test, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residuals vs. Predicted Values on Test Set'); plt.xlabel('Predicted'); plt.ylabel('Residuals')
        plt.savefig(PLOTS_DIR / "residuals_plot.png"); plt.close()

def main():
    """Loads test data and runs the evaluation of the best models."""
    # --- 1. Load Data ---
    test_df = pd.read_csv(TEST_PATH)
    test_df[TARGET_CLASSIFICATION] = test_df[TARGET_CLASSIFICATION].map({'Yes': 1, 'No': 0})
    
    X_test = test_df.drop(columns=DROP_FEATURES + [TARGET_CLASSIFICATION, TARGET_REGRESSION])
    y_test_clf = test_df[TARGET_CLASSIFICATION]
    y_test_reg = test_df[TARGET_REGRESSION]

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # --- 2. Evaluate Models using the Helper Function ---
    _evaluate_best_pipeline(
        task_type="classification", X_test=X_test, y_test=y_test_clf,
        filter_string="params.model_type = 'classification'",
        order_by="metrics.val_f1_score DESC"
    )
    
    _evaluate_best_pipeline(
        task_type="regression", X_test=X_test, y_test=y_test_reg,
        filter_string="params.model_type = 'regression'",
        order_by="metrics.val_rmse ASC"
    )

    logging.info("--- Final Evaluation Complete ---")


if __name__ == "__main__":
    main()