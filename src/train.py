from src.config import MODEL_PATH, MODELS_DIR
from src.data import load_data
from src.model import get_model

import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score

def main():
    # Data load
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    # Model creation
    print("Creating model...")
    model = get_model()

    # Model training
    print("Training model...")
    model.fit(X_train, y_train)

    # Model evaluation
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_pr = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_pr)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # Save model
    print("Saving model...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Model saved at {MODEL_PATH}")

if __name__ == "__main__":
    main()