from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "diabetes_prediction_dataset.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"

# Data
TARGET_COL = "diabetes"

CATEGORICAL_COLS = [
    "gender",
    "smoking_history",
]

NUMERICAL_COLS = [
    "age",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "hypertension",
    "heart_disease",
]

# Train / test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model
MODEL_TYPE = "logistic_regression"

LOGREG_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}

# Evaluation
METRICS = ["accuracy", "roc_auc"]
