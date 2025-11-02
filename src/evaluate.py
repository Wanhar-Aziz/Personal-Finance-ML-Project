import logging
from typing import List

import pandas as pd
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score,
                             mean_absolute_error, root_mean_squared_error)

from src.config import (
    TEST_PATH, MLFLOW_EXPERIMENT_NAME, TARGET_CLASSIFICATION, TARGET_REGRESSION,
    PLOTS_DIR, TABLES_DIR, DROP_FEATURES
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _evaluate_all_pipelines(
    task_type: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    filter_string: str,
    order_by: str
) -> None:
    """Finds all relevant pipelines in MLflow, evaluates them, saves comprehensive
    metrics, and plots artifacts for the best one."""
    logging.info(f"--- Evaluating all {task_type} pipelines ---")
    
    all_runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT_NAME],
        filter_string=filter_string,
        order_by=[order_by]
        # Note: We removed max_results=1 to get all runs
    )

    if all_runs.empty:
        logging.warning(f"No {task_type} models found. Skipping.")
        return

    all_metrics = []
    for index, row in all_runs.iterrows():
        run_id = row["run_id"]
        model_name = row['tags.mlflow.runName'].split('_')[0]
        model_uri = f"runs:/{run_id}/{model_name}"
        
        logging.info(f"Loading and evaluating: {model_name} (run ID: {run_id})")
        pipeline = mlflow.sklearn.load_model(model_uri)
        y_pred_test = pipeline.predict(X_test)
        
        # This is the best model in the sorted list, so we'll generate plots for it
        is_best_model = (index == 0)

        if task_type == "classification":
            accuracy = accuracy_score(y_test, y_pred_test)
            f1 = f1_score(y_test, y_pred_test, average='weighted')


            # 4 decimal places for classification metrics
            accuracy = round(accuracy, 4)
            f1 = round(f1, 4)

            all_metrics.append({"Model": model_name, "Accuracy": accuracy, "F1 Score": f1})

            if is_best_model:
                logging.info(f"  -> Best Model. Test F1: {f1:.4f}. Generating confusion matrix.")
                cm = confusion_matrix(y_test, y_pred_test)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
                plt.title(f'Confusion Matrix on Test Set ({model_name})'); plt.xlabel('Predicted'); plt.ylabel('Actual')
                plt.savefig(PLOTS_DIR / "confusion_matrix.png"); plt.close()
        
        elif task_type == "regression":
            mae = mean_absolute_error(y_test, y_pred_test)
            rmse = root_mean_squared_error(y_test, y_pred_test)

            # 2 decimal places for regression metrics
            mae = round(mae, 2)
            rmse = round(rmse, 2)

            all_metrics.append({"Model": model_name, "MAE": mae, "RMSE": rmse})

            if is_best_model:
                logging.info(f"  -> Best Model. Test RMSE: {rmse:.4f}. Generating residuals plot.")
                residuals = y_test - y_pred_test
                plt.figure(figsize=(8, 6))
                plt.scatter(y_pred_test, residuals, alpha=0.5)
                plt.axhline(y=0, color='r', linestyle='--')
                plt.title(f'Residuals vs. Predicted Values ({model_name})'); plt.xlabel('Predicted'); plt.ylabel('Residuals')
                plt.savefig(PLOTS_DIR / "residuals_plot.png"); plt.close()

    # Save the comprehensive metrics table
    metrics_df = pd.DataFrame(all_metrics).set_index("Model")
    output_path = TABLES_DIR / f"{task_type}_test_metrics.csv"
    metrics_df.to_csv(output_path)
    logging.info(f"Saved comprehensive {task_type} test metrics to {output_path}")
    print(metrics_df)


def main():
    """Loads test data and orchestrates the evaluation of all baseline models."""
    test_df = pd.read_csv(TEST_PATH)
    test_df[TARGET_CLASSIFICATION] = test_df[TARGET_CLASSIFICATION].map({'Yes': 1, 'No': 0})
    
    X_test = test_df.drop(columns=DROP_FEATURES + [TARGET_CLASSIFICATION, TARGET_REGRESSION])
    y_test_clf = test_df[TARGET_CLASSIFICATION]
    y_test_reg = test_df[TARGET_REGRESSION]

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    _evaluate_all_pipelines(
        task_type="classification", X_test=X_test, y_test=y_test_clf,
        filter_string="params.model_type = 'classification'",
        order_by="metrics.val_f1_score DESC"
    )
    
    _evaluate_all_pipelines(
        task_type="regression", X_test=X_test, y_test=y_test_reg,
        filter_string="params.model_type = 'regression'",
        order_by="metrics.val_rmse ASC"
    )

    logging.info("--- Final Evaluation Complete ---")


if __name__ == "__main__":
    main()