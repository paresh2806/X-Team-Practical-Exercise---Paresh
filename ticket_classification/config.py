from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "v2_tfidf_logreg_balanced.joblib"

TEXT_COLUMN = "text"
PREDICTION_COLUMN = "predicted_label"

ALLOWED_LABELS = {
    "account-access",
    "transaction-dispute",
    "fraud-report",
    "general",
}