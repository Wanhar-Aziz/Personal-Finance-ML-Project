# In src/visualization.py
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import PLOTS_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def plot_class_distribution(
    data: pd.DataFrame,
    column: str,
    output_path: Path
) -> None:
    """Plots and saves the distribution of a categorical column.

    Args:
        data: The DataFrame containing the data.
        column: The name of the categorical column to plot.
        output_path: The path to save the output plot image.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x=column, data=data, palette=["steelblue", "salmon"], hue=column)
    plt.title(f"Class Distribution of '{column}'", fontsize=16)
    plt.xlabel(column, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    logging.info(f"Class distribution plot saved to {output_path}")
    plt.close()


def plot_correlation_heatmap(
    data: pd.DataFrame,
    output_path: Path
) -> None:
    """Plots and saves a correlation heatmap for the numeric features of a DataFrame.

    Args:
        data: The DataFrame containing the data.
        output_path: The path to save the output plot image.
    """
    numeric_df = data.select_dtypes(include=np.number)
    corr = numeric_df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5)
    plt.title("Correlation Heatmap of Numeric Features", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    logging.info(f"Correlation heatmap saved to {output_path}")
    plt.close()
