import os
from pathlib import Path

# CORE PATHS
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "synthetic_personal_finance_dataset.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
REPORTS_DIR = BASE_DIR / "outputs" / "reports"

# COLLABORATION
TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_PATH = PROCESSED_DATA_DIR / "test.csv"

# ML PARAMETERS
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2
KAGGLE_DATASET = "miadul/personal-finance-ml-dataset"

# TARGET VARIABLES
TARGET_CLASSIFICATION = "has_loan"
TARGET_REGRESSION = "credit_score"

# MLFLOW CONFIG
MLFLOW_EXPERIMENT_NAME = "personal-finance-baselines"

# FEATURE ENGINEERING & SELECTION
CATEGORICAL_FEATURES = ['gender', 'education_level', 'employment_status', 'job_title', 'region']
DROP_FEATURES = ['user_id', 'record_date']

# Features to remove for classification due to data leakage
CLASSIFICATION_DROP_COLS = [
    'loan_amount_usd', 'loan_term_months', 'monthly_emi_usd', 
    'loan_interest_rate_pct', 'loan_type', 'debt_to_income_ratio'
]

# Features to remove for regression (only loan_type due to its high missingness before imputation)
REGRESSION_DROP_COLS = ['loan_type']
