# **Personal Finance ML Project**

![Python 3.11.13](https://img.shields.io/badge/python-3.11.13-blue.svg)

This repository contains the complete pipeline for a machine learning project for the CS-4120 course. The project tackles two key prediction tasks using a synthetic Personal Finance ML Dataset from Kaggle:

1.  **Classification:** Predict loan eligibility (`has_loan`).
2.  **Regression:** Predict an individual's credit score (`credit_score`).

The project follows MLOps best practices, including a structured, modular codebase, a reproducible environment, and experiment tracking with MLflow.

---

## **Project Structure**

The repository is organized to separate concerns, making the pipeline modular and easy to maintain.

```
Personal-Finance-ML-Project/
├── .gitignore              # Specifies files and directories to be ignored by Git.
├── .python-version         # Pins the project's Python version for pyenv.
├── README.md               # This file.
├── requirements.txt        # A list of all Python package dependencies.
├── data/
│   ├── raw/                # (Gitignored) The original dataset is stored here after download.
│   └── processed/          # (Gitignored) The cleaned, split data (train/val/test) is saved here.
├── outputs/
│   ├── plots/              # EDA and evaluation plots are saved here.
│   ├── tables/             # Evaluation tables are saved here.
│   └── reports/            # Generated PDF reports are saved here.
├── scripts/
│   ├── eda.py              # Convenience script to run only the EDA part of the pipeline.
│   ├── make_dataset.py     # Convenience script to run only the data processing part.
│   ├── report_generator.py # Convenience script to generate the PDF report.
│   └── run_pipeline.py     # Convenience script to run the main pipeline from the root.
└── src/
    ├── config.py           # Central configuration for paths, parameters, and seeds.
    ├── data_processing.py  # Script for data loading, cleaning, and splitting.
    ├── visualization.py    # Script for generating all plots.
    ├── report.py           # Script for generating the PDF report.
    ├── train_baselines.py  # Script for training classical ML models with MLflow.
    ├── evaluate.py         # Script for evaluating the best models on the test set.
    └── main.py             # Main orchestrator script with a command-line interface (CLI).
```

---

## **Getting Started**

Follow these steps to set up and run the project locally.

### **Prerequisites**

*   **Git:** To clone the repository.
*   **Python 3.11.13:** The project is pinned to this specific version. We strongly recommend using `pyenv` to manage Python versions.
*   **Kaggle Account & API Credentials:** For automatic dataset download.

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/Wanhar-Aziz/Personal-Finance-ML-Project.git
cd Personal-Finance-ML-Project
```

### **Step 2: Authenticate with Kaggle**

The pipeline uses the `kagglehub` library to download the dataset automatically. This requires you to authenticate your machine with your Kaggle account. Please refer to the official Kaggle authentication guide: **[CLICK HERE](https://www.kaggle.com/discussions/getting-started/524433)**.

### **Step 3: Set Up the Python Environment**

This project uses a virtual environment to manage dependencies.

1.  **(Recommended) Set the Python Version:** If you use `pyenv`, it will automatically detect the `.python-version` file.

2.  **Create and Activate the Virtual Environment:**
    ```bash
    python3 -m venv venv
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies and the Project Package:**
    ```bash
    # Install all required libraries
    pip install -r requirements.txt
    
    # Install the project's src/ folder as an editable package
    pip install -e .
    ```
---

## **Running the Pipeline**

The entire pipeline is processed by `src/main.py`, which provides a command-line interface.

### **Basic Run**
This will process the data and prepare it for modeling. It will download the dataset if it's not found locally.

```bash
python src/main.py
```

### **Running with EDA**
To also generate and save the EDA plots:

```bash
python src/main.py --run-eda
```

### **Forcing a Fresh Download**
To force the pipeline to re-download the dataset from Kaggle, even if a local copy exists:
```bash
python src/main.py --force-download
```

The full pipeline will:
1.  **Automatically download** the dataset from Kaggle and save it to `data/raw/`.
2.  Process the raw data and save train, validation, and test sets to `data/processed/`.
3.  (Soon) Train the classical baseline models and log results to MLflow.
4.  (Soon) Evaluate the best models and save artifacts to `outputs/`.
5.  **Generate a PDF report** summarizing the data processing steps.

---

## **Experiment Tracking with MLflow**

This project uses MLflow to track experiments. To view the results:

1.  Make sure your virtual environment is activated.
2.  Run the MLflow UI from the project root:
    ```bash
    mlflow ui
    ```
3.  Open your web browser and navigate to `http://localhost:5000` to see all your experiment runs, metrics, and saved models.
