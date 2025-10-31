import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.report import generate_report

if __name__ == "__main__":
    generate_report()