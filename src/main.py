import argparse
from pathlib import Path

from src.config import RAW_DATA_PATH
from src.data_processing import clean_data, download_dataset, load_data, split_data
from src.report import generate_report


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
    ensure_dataset(force_download=force_download)

    df = clean_data(load_data())
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

    generate_report()

    print("Pipeline complete.")
    print(f"Total rows: {len(df)}")
    print(f"Train rows: {len(train_df)} | Validation rows: {len(val_df)} | Test rows: {len(test_df)}")
    print("Processed data saved to data/processed/. Visuals and reports saved under outputs/.")


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
