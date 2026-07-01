import pandas as pd
import pytest

from ticket_classification.config import PREDICTION_COLUMN
from ticket_classification.score_holdout import score_holdout


def test_score_holdout_requires_existing_input_file(tmp_path):
    input_path = tmp_path / "missing.csv"
    output_path = tmp_path / "predictions.csv"

    with pytest.raises(FileNotFoundError):
        score_holdout(str(input_path), str(output_path))


def test_score_holdout_requires_text_column(tmp_path):
    input_path = tmp_path / "bad_input.csv"
    output_path = tmp_path / "predictions.csv"

    df = pd.DataFrame(
        {
            "message": ["I cannot login to my account"]
        }
    )
    df.to_csv(input_path, index=False)

    with pytest.raises(ValueError):
        score_holdout(str(input_path), str(output_path))


def test_score_holdout_writes_predictions(tmp_path):
    input_path = tmp_path / "holdout.csv"
    output_path = tmp_path / "predictions.csv"

    df = pd.DataFrame(
        {
            "text": [
                "I cannot login to my account",
                "There is an unauthorized transaction on my wallet",
                "My card was charged twice for the same transaction",
            ]
        }
    )
    df.to_csv(input_path, index=False)

    score_holdout(str(input_path), str(output_path))

    result = pd.read_csv(output_path)

    assert output_path.exists()
    assert PREDICTION_COLUMN in result.columns
    assert len(result) == 3
    assert result[PREDICTION_COLUMN].notna().all()