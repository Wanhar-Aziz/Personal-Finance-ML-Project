import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.data_processing import download_dataset, load_data, clean_data, split_data

if __name__ == "__main__":
    raw_path = download_dataset()
    df = load_data()
    df = clean_data(df)
    split_data(df)