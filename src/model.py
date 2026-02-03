from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from src.config import (
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    LOGREG_PARAMS,
)


def get_model() -> Pipeline:
    """
    Create a pipeline that includes preprocessing and a Logistic Regression classifier.
    """
    # Preprocessing for numerical features
    numerical_transformer = StandardScaler()

    # Preprocessing for categorical features 
    # (we observed at EDA that these variables have more that 2 unique values, so we use One-Hot Encoder)
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    # Combination of preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ]
    )

    # Model definition
    model = LogisticRegression(**LOGREG_PARAMS)

    # Pipeline creation
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    return pipeline
