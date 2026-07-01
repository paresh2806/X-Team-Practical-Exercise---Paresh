import subprocess
import sys

import pandas as pd


def test_cli_predict_command_returns_label():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ticket_classification.cli",
            "predict",
            "I cannot login to my account",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    prediction = result.stdout.strip()

    assert prediction in {
        "account-access",
        "transaction-dispute",
        "fraud-report",
        "general",
    }


def test_cli_score_command_writes_predictions(tmp_path):
    input_path = tmp_path / "holdout.csv"
    output_path = tmp_path / "predictions.csv"

    df = pd.DataFrame(
        {
            "text": [
                "I cannot login to my account",
                "There is an unauthorized transaction on my wallet",
            ]
        }
    )
    df.to_csv(input_path, index=False)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "ticket_classification.cli",
            "score",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    result = pd.read_csv(output_path)

    assert output_path.exists()
    assert "predicted_label" in result.columns
    assert len(result) == 2