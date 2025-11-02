import argparse
from pathlib import Path
import logging

import pandas as pd

from src.data_processing import clean_data, download_dataset, load_data, split_data, engineer_features
from src.evaluate import main as evaluate_main
from src.train_baselines import main as train_baselines_main
from src.visualization import plot_class_distribution, plot_correlation_heatmap
from src.report import generate_report

from src.config import (
    KAGGLE_DATASET,
    PLOTS_DIR, 
    PROCESSED_DATA_DIR, 
    RAW_DATA_DIR,
    RAW_DATA_PATH, 
    RANDOM_STATE,
    TARGET_CLASSIFICATION, 
    TEST_SIZE, 
    VAL_SIZE
    )

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_dataset(force_download: bool) -> Path:
    """
    Make sure a raw dataset exists locally.
    Downloads from Kaggle when forced or when the cached copy is absent.
    """
    # Check if file exists already and not forcing download
    if RAW_DATA_PATH.exists() and not force_download:
        logging.info(f"Using cached dataset at {RAW_DATA_PATH}")
        return RAW_DATA_PATH

    try:
        logging.info("Fetching dataset from Kaggle...")
        return download_dataset(KAGGLE_DATASET, RAW_DATA_DIR)
    except Exception as exc:  # pragma: no cover - provides helpful context for runtime failures
        logging.error(f"Error occurred while downloading dataset: {exc}")
        raise RuntimeError(
            "Unable to download the dataset. "
            "Confirm Kaggle credentials are configured or manually download the dataset and place it at "
            f"{RAW_DATA_PATH}."
        ) from exc

def _run_data_prep_stage(force_download: bool) -> None:
    """Handles the data preparation stage of the pipeline."""
    logging.info("--- STAGE 1: DATA PREPARATION ---")
    ensure_dataset(force_download=force_download)

    df = load_data(path=RAW_DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df)
    train_df, val_df, test_df = split_data(
        df=df,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        output_dir=PROCESSED_DATA_DIR
    )
    logging.info(f"Train rows: {len(train_df)} | Validation rows: {len(val_df)} | Test rows: {len(test_df)}")
    logging.info("Data preparation complete.")

def _run_eda_stage() -> None:
    """Handles the optional EDA stage of the pipeline."""
    logging.info("--- (Optional) STAGE: EXPLORATORY DATA ANALYSIS ---")
    # Load the full, processed dataset before splitting to run EDA
    df = pd.read_csv(RAW_DATA_DIR / RAW_DATA_PATH.name)
    plot_class_distribution(
        data=df,
        column=TARGET_CLASSIFICATION,
        output_path=PLOTS_DIR / "class_distribution.png"
    )
    plot_correlation_heatmap(
        data=df,
        output_path=PLOTS_DIR / "correlation_heatmap.png"
    )
    logging.info("EDA complete. Plots saved to outputs/plots/.")

def run_pipeline(force_download: bool = False, run_eda: bool = False) -> None:
    """Execute the end-to-end data preparation pipeline with optional EDA visuals."""
    logging.info("--- STAGE 1: DATA PREPARATION ---")

    ensure_dataset(force_download=force_download)

    df = load_data(path=RAW_DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df)
    train_df, val_df, test_df = split_data(
        df=df,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        output_dir=PROCESSED_DATA_DIR
    )

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
        help="Generate EDA plots.",
    )
    return parser

def main(args: argparse.Namespace) -> None:
    """Main function to run the ML pipeline."""
    _run_data_prep_stage(force_download=args.force_download)

    if args.run_eda:
        _run_eda_stage()

    logging.info("--- STAGE 2: TRAINING BASELINE MODELS ---")
    train_baselines_main()

    logging.info("--- STAGE 3: FINAL EVALUATION ---")
    evaluate_main()
    
    logging.info("--- STAGE 4: REPORT GENERATION ---")
    generate_report()

    logging.info("Pipeline complete.")

# def main(argv: list[str] | None = None) -> None:
#     args = build_arg_parser().parse_args(argv)
#     run_pipeline(force_download=args.force_download, run_eda=args.run_eda)


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    main(args)
