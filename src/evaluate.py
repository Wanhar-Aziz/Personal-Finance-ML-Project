# In src/evaluate.py
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


def main():
    """
    Loads the best pipeline from MLflow for each task, evaluates it on the test set,
    and saves the final artifacts.
    """
    logging.info("--- Starting Final Evaluation on Test Set ---")

    # Load test data and prepare targets
    test_df = pd.read_csv(TEST_PATH)
    test_df[TARGET_CLASSIFICATION] = test_df[TARGET_CLASSIFICATION].map({'Yes': 1, 'No': 0})
    
    y_test_clf = test_df[TARGET_CLASSIFICATION]
    y_test_reg = test_df[TARGET_REGRESSION]
    X_test = test_df.drop(columns=DROP_FEATURES + [TARGET_CLASSIFICATION, TARGET_REGRESSION])

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # --- 1. Evaluate Best Classification Model ---
    logging.info("Evaluating best classification pipeline...")
    # Find the best run by looking for the highest validation F1 score
    best_clf_runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT_NAME],
        filter_string="params.model_type = 'classification'",
        order_by=["metrics.val_f1_score DESC"],
        max_results=1
    )

    if best_clf_runs.empty:
        logging.warning("No classification models found. Skipping.")
    else:
        best_clf_run_id = best_clf_runs.iloc[0]["run_id"]
        model_name = best_clf_runs.iloc[0]['tags.mlflow.runName'].split('_')[0]
        model_uri = f"runs:/{best_clf_run_id}/{model_name}"
        
        logging.info(f"Loading best classification pipeline from run: {best_clf_run_id}")
        best_clf_pipeline = mlflow.sklearn.load_model(model_uri)
        
        # Make predictions on raw test data
        y_pred_test_clf = best_clf_pipeline.predict(X_test)
        
        # Calculate and log metrics
        test_accuracy = accuracy_score(y_test_clf, y_pred_test_clf)
        test_f1 = f1_score(y_test_clf, y_pred_test_clf, average='weighted')
        
        logging.info(f"  Test Accuracy: {test_accuracy:.4f}")
        logging.info(f"  Test F1-Score: {test_f1:.4f}")
        
        # Save metrics table and confusion matrix plot
        clf_metrics = pd.DataFrame({"Metric": ["Accuracy", "F1-Score"], "Value": [test_accuracy, test_f1]})
        clf_metrics.to_csv(TABLES_DIR / "classification_test_metrics.csv", index=False)
        
        cm = confusion_matrix(y_test_clf, y_pred_test_clf)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        plt.title('Confusion Matrix on Test Set')
        plt.xlabel('Predicted'); plt.ylabel('Actual')
        plt.savefig(PLOTS_DIR / "confusion_matrix.png")
        plt.close()

    # --- 2. Evaluate Best Regression Model ---
    logging.info("Evaluating best regression pipeline...")
    # Find the best run by looking for the lowest validation RMSE
    best_reg_runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT_NAME],
        filter_string="params.model_type = 'regression'",
        order_by=["metrics.val_rmse ASC"],
        max_results=1
    )

    if best_reg_runs.empty:
        logging.warning("No regression models found. Skipping.")
    else:
        best_reg_run_id = best_reg_runs.iloc[0]["run_id"]
        model_name = best_reg_runs.iloc[0]['tags.mlflow.runName'].split('_')[0]
        model_uri = f"runs:/{best_reg_run_id}/{model_name}"

        logging.info(f"Loading best regression pipeline from run: {best_reg_run_id}")
        best_reg_pipeline = mlflow.sklearn.load_model(model_uri)
        
        # Make predictions on raw test data
        y_pred_test_reg = best_reg_pipeline.predict(X_test)
        
        # Calculate and log metrics
        test_mae = mean_absolute_error(y_test_reg, y_pred_test_reg)
        test_rmse = root_mean_squared_error(y_test_reg, y_pred_test_reg)
        
        logging.info(f"  Test MAE: {test_mae:.4f}")
        logging.info(f"  Test RMSE: {test_rmse:.4f}")

        # Save metrics table and residuals plot
        reg_metrics = pd.DataFrame({"Metric": ["MAE", "RMSE"], "Value": [test_mae, test_rmse]})
        reg_metrics.to_csv(TABLES_DIR / "regression_test_metrics.csv", index=False)
        
        residuals = y_test_reg - y_pred_test_reg
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred_test_reg, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residuals vs. Predicted Values on Test Set')
        plt.xlabel('Predicted Credit Score'); plt.ylabel('Residuals')
        plt.savefig(PLOTS_DIR / "residuals_plot.png")
        plt.close()

    logging.info("--- Final Evaluation Complete ---")


if __name__ == "__main__":
    main()