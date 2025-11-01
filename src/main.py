import argparse
from pathlib import Path
import logging

from src.data_processing import clean_data, download_dataset, load_data, split_data, engineer_features
from src.train_baselines import main as train_baselines_main
from src.evaluate import main as evaluate_main
from src.report import generate_report
from src.config import RAW_DATA_PATH

# TODO: Configure logging more granularly in each module
# TODO: Move logging configuration to a dedicated logging module
# TODO: Improve documentation throughout the codebase
# TODO: Add type hints throughout the codebase
# TODO: Avoid monolithic functions; break into smaller functions where appropriate
# TODO: Improve report generation with more insights and visualizations
# TODO: move pkl files to a more appropriate location (maybe artifacts/ or models/)
# TODO: Ensure MLflow runs are properly tagged and organized
# TODO: Confirm that data processing handles all edge cases in the dataset
# TODO: Check outliers handling in data cleaning
# TODO: streamline the pipeline for efficiency
# TODO: Implement model improvement strategies such as hyperparameter tuning
# IDEAS: Add command-line arguments for more pipeline options

# NEED: Add standard deviation for metrics where applicable

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_dataset(force_download: bool) -> Path:
    """
    Make sure a raw dataset exists locally.
    Downloads from Kaggle when forced or when the cached copy is absent.
    """
    if RAW_DATA_PATH.exists() and not force_download:
        return RAW_DATA_PATH

    try:
        print("Fetching dataset from Kaggle...")
        return download_dataset()
    except Exception as exc:  # pragma: no cover - provides helpful context for runtime failures
        raise RuntimeError(
            "Unable to download the dataset. "
            "Confirm Kaggle credentials are configured or place the dataset at "
            f"{RAW_DATA_PATH}."
        ) from exc


def run_pipeline(force_download: bool = False, run_eda: bool = False) -> None:
    """Execute the end-to-end data preparation pipeline with optional EDA visuals."""
    logging.info("--- STAGE 1: DATA PREPARATION ---")

    ensure_dataset(force_download=force_download)

    df = clean_data(load_data())
    df = engineer_features(df)
    train_df, val_df, test_df = split_data(df)

    if run_eda:
        try:
            from src.visualization import plot_class_distribution, plot_correlation_heatmap
        except Exception as exc:
            print(f"Skipping EDA visualizations due to import error: {exc}")
        else:
            plot_class_distribution(df)
            plot_correlation_heatmap(df)
    else:
        print("Skipping EDA visualizations. Use --run-eda to enable.")

    logging.info(f"Total rows: {len(df)}")
    logging.info(f"Train rows: {len(train_df)} | Validation rows: {len(val_df)} | Test rows: {len(test_df)}")
    logging.info("Processed data saved to data/processed/. Visuals and reports saved under outputs/.")

    logging.info("--- STAGE 2: TRAINING BASELINE MODELS ---")
    train_baselines_main()

    logging.info("--- STAGE 3: FINAL EVALUATION ---")
    evaluate_main()

    logging.info("--- STAGE 4: REPORT GENERATION ---")
    generate_report()

    logging.info("Pipeline complete.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Personal Finance ML end-to-end preprocessing pipeline."
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Always download the dataset from Kaggle even if a local copy exists.",
    )
    parser.add_argument(
        "--run-eda",
        action="store_true",
        help="Generate EDA plots (requires matplotlib).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_pipeline(force_download=args.force_download, run_eda=args.run_eda)


if __name__ == "__main__":
    main()
