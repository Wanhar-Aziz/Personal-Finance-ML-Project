import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from kagglehub import dataset_download
from src.config import (
    RAW_DATA_DIR,
    RAW_DATA_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    VAL_SIZE,
    KAGGLE_DATASET,
)

def download_dataset():
    """Downloads the latest version of the Kaggle dataset using kagglehub."""
    path = dataset_download(KAGGLE_DATASET)
    print("Path to dataset files:", path)
    df = pd.read_csv(f"{path}/synthetic_personal_finance_dataset.csv")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Dataset saved to {RAW_DATA_PATH}")
    return RAW_DATA_PATH

def load_data():
    """Loads dataset from local path."""
    return pd.read_csv(RAW_DATA_PATH)

def clean_data(df):
    """Handles missing values in the loan_type column."""
    if "loan_type" in df.columns:
        df["loan_type"] = df["loan_type"].fillna("Missing")
    return df

def split_data(df):
    """Splits the dataset reproducibly into train, validation, and test."""
    train_val_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    val_proportion = VAL_SIZE / (1 - TEST_SIZE)

    train_df, val_df = train_test_split(train_val_df, test_size=val_proportion, random_state=RANDOM_STATE)
    
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    print("Train/Val/Test splits saved to data/processed/")
    
    return train_df, val_df, test_df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates new features to improve model performance.

    Args:
        df: The input DataFrame.

    Returns:
        The DataFrame with new features added.
    """
    # Use a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Define a small epsilon to prevent division by zero
    epsilon = 1e-6

    # 1. Expenses-to-Income Ratio
    df['expenses_to_income_ratio'] = df['monthly_expenses_usd'] / (df['monthly_income_usd'] + epsilon)

    # 2. Savings as a Multiple of Monthly Expenses
    df['savings_as_multiple_of_expenses'] = df['savings_usd'] / (df['monthly_expenses_usd'] + epsilon)
    
    # Handle potential infinite values resulting from division by a near-zero number
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Impute any resulting NaNs with the column median
    df['expenses_to_income_ratio'] = df['expenses_to_income_ratio'].fillna(df['expenses_to_income_ratio'].median())
    df['savings_as_multiple_of_expenses'] = df['savings_as_multiple_of_expenses'].fillna(df['savings_as_multiple_of_expenses'].median())

    print("Engineered 2 new features: 'expenses_to_income_ratio' and 'savings_as_multiple_of_expenses'.")
    return df

if __name__ == "__main__":
    raw_path = download_dataset()
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    split_data(df)
