import joblib

from ticket_classification.config import ALLOWED_LABELS, MODEL_PATH


_model = None


def validate_text(text: str) -> str:
    """
    Validate a single input message before prediction.

    Rules:
    - input must be a string
    - input cannot be empty
    - input cannot be whitespace only
    """
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")

    cleaned_text = text.strip()

    if not cleaned_text:
        raise ValueError("Input text cannot be empty.")

    return cleaned_text


def load_model():
    """
    Load the trained sklearn Pipeline from disk.

    The saved pipeline includes both:
    - TF-IDF vectorizer
    - Logistic Regression classifier
    """
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

        _model = joblib.load(MODEL_PATH)

    return _model


def predict(text: str) -> str:
    """
    Predict the support-ticket route for a single input message.

    Returns one of:
    - account-access
    - transaction-dispute
    - fraud-report
    - general
    """
    cleaned_text = validate_text(text)
    model = load_model()

    prediction = model.predict([cleaned_text])[0]

    if prediction not in ALLOWED_LABELS:
        raise ValueError(f"Unexpected model prediction: {prediction}")

    return prediction