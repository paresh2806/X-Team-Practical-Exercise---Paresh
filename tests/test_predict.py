import pytest

from ticket_classification.config import ALLOWED_LABELS
from ticket_classification.predict import predict, validate_text


def test_validate_text_accepts_valid_text():
    text = "I cannot login to my account"
    assert validate_text(text) == text


def test_validate_text_strips_whitespace():
    text = "   I cannot login to my account   "
    assert validate_text(text) == "I cannot login to my account"


def test_validate_text_rejects_empty_string():
    with pytest.raises(ValueError):
        validate_text("")


def test_validate_text_rejects_whitespace_only_string():
    with pytest.raises(ValueError):
        validate_text("   ")


def test_validate_text_rejects_non_string():
    with pytest.raises(ValueError):
        validate_text(None)


def test_predict_returns_allowed_label():
    prediction = predict("I cannot login to my account")
    assert prediction in ALLOWED_LABELS