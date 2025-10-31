from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.config import PLOTS_DIR, TARGET_CLASSIFICATION


ArtifactLogger = Optional[Callable[[str], None]]


def _log_artifact(path, artifact_logger: ArtifactLogger) -> None:
    if artifact_logger is not None:
        artifact_logger(str(path))


def plot_class_distribution(df, artifact_logger: ArtifactLogger = None):
    plt.figure(figsize=(6,4))
    df[TARGET_CLASSIFICATION].value_counts().plot(kind="bar", color=["steelblue", "salmon"])
    plt.title(f"Class Distribution: {TARGET_CLASSIFICATION}")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    path = PLOTS_DIR / "class_distribution_has_loan.png"
    plt.savefig(path, dpi=200)
    _log_artifact(path, artifact_logger)
    plt.close()


def plot_correlation_heatmap(df, artifact_logger: ArtifactLogger = None):
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()
    plt.figure(figsize=(7,6))
    plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
    plt.colorbar()
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.tight_layout()
    path = PLOTS_DIR / "correlation_heatmap.png"
    plt.savefig(path, dpi=200)
    _log_artifact(path, artifact_logger)
    plt.close()
