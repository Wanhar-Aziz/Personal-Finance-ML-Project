from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_preprocessing_pipeline(
    numeric_features: List[str], categorical_features: List[str]
) -> Pipeline:
    """Creates a Scikit-learn pipeline for preprocessing data.

    The pipeline scales numeric features and one-hot encodes categorical features.

    Args:
        numeric_features: A list of column names for numeric features.
        categorical_features: A list of column names for categorical features.

    Returns:
        A Scikit-learn Pipeline object.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",  # Keep other columns if any; can be set to 'drop'
    )

    return Pipeline(steps=[("preprocessor", preprocessor)])