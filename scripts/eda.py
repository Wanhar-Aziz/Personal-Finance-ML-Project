import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.data_processing import load_data, clean_data

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    try:
        from src.visualization import plot_class_distribution, plot_correlation_heatmap
    except Exception as exc:
        print(f"Unable to import visualization utilities: {exc}")
    else:
        plot_class_distribution(df)
        plot_correlation_heatmap(df)
        print("EDA complete. Plots saved to outputs/plots/")
