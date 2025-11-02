import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from kagglehub import dataset_download
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_dataset(kaggle_dataset: str, output_dir: Path):
    """Downloads a dataset from Kaggle and saves it to a specified directory.

    Args:
        kaggle_dataset: The Kaggle dataset identifier (e.g., 'user/dataset-name').
        output_dir: The directory where the raw dataset CSV will be saved.

    Returns:
        The file path to the downloaded dataset CSV.
    """
    logging.info(f"Downloading dataset '{kaggle_dataset}' from Kaggle...")

    # kagglehub downloads to a cache; we'll read from there and save to our structured path
    path = dataset_download(kaggle_dataset)
    logging.info("Path to dataset files: %s", path)

    # The actual CSV name might vary, so we find the first CSV in the downloaded folder
    csv_files = list(Path(path).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in the downloaded dataset at {path}")
    
    source_csv_path = csv_files[0]
    df = pd.read_csv(source_csv_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_dir / source_csv_path.name
    df.to_csv(output_csv_path, index=False)
    logging.info(f"Dataset saved to {output_csv_path}")
    return output_csv_path

def load_data(path: Path) -> pd.DataFrame:
    """Loads a dataset from a local CSV file path.

    Args:
        path: The file path to the CSV data.

    Returns:
        A pandas DataFrame containing the loaded data.
    """
    logging.info(f"Loading data from {path}...")
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Applies basic cleaning steps to the DataFrame.

    Currently, this function handles missing values in the 'loan_type' column
    by filling them with the string 'Missing'.

    Args:
        df: The input pandas DataFrame.

    Returns:
        The cleaned pandas DataFrame.
    """
    df = df.copy()
    if "loan_type" in df.columns:
        df["loan_type"] = df["loan_type"].fillna("Missing")
    logging.info("Data cleaning complete.")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates new features to improve model performance.

    Args:
        df: The input DataFrame.

    Returns:
        The DataFrame with new features added.
    """
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

    logging.info("Engineered 2 new features: 'expenses_to_income_ratio' and 'savings_as_multiple_of_expenses'.")
    return df

def split_data(
        df: pd.DataFrame, test_size: float, val_size: float, random_state: int, output_dir: Path
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the dataset into train, validation, and test sets and saves them.

    Args:
        df: The DataFrame to split.
        test_size: The proportion of the dataset to allocate to the test split.
        val_size: The proportion of the dataset to allocate to the validation split.
        random_state: The seed for the random number generator for reproducibility.
        output_dir: The directory where the split data CSVs will be saved.

    Returns:
        A tuple containing the train, validation, and test DataFrames.
    """
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    val_proportion = val_size / (1 - test_size)
    train_df, val_df = train_test_split(train_val_df, test_size=val_proportion, random_state=random_state)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    logging.info(f"Train/Val/Test splits saved to {output_dir}")
    return train_df, val_df, test_df


if __name__ == "__main__":
    raw_path = download_dataset()
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    split_data(df)
