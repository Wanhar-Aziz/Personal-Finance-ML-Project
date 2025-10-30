import os
from pathlib import Path

# CORE PATHS
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "NAME_OF_DATASET.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
TABLES_DIR = BASE_DIR / "outputs" / "tables"

# COLLABORATION
TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_PATH = PROCESSED_DATA_DIR / "test.csv"

# ML PARAMETERS
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2

# TARGET VARIABLES
TARGET_CLASSIFICATION = "has_loan"
TARGET_REGRESSION = "credit_score"

# MLFLOW CONFIG
MLFLOW_EXPERIMENT_NAME = "personal-finance-baselines"

# Ensure output directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
