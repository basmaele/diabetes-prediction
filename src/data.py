import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

from src.config import (
    RAW_DATA_PATH,
    TARGET_COL,
    TEST_SIZE,
    RANDOM_STATE,
)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load the diabetes dataset, split it into train and test sets,
    and separate features from target.
    """
    # Load raw data
    df = pd.read_csv(RAW_DATA_PATH)

    # Separate features and target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test
