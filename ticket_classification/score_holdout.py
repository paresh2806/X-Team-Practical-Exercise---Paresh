import argparse
from pathlib import Path

import pandas as pd

from ticket_classification.config import PREDICTION_COLUMN, TEXT_COLUMN
from ticket_classification.predict import predict


def score_holdout(input_path: str, output_path: str) -> None:
    """
    Read a CSV containing support-ticket messages and write predictions.

    Expected input CSV:
    - must contain a 'text' column

    Output CSV:
    - contains all original columns
    - adds a 'predicted_label' column
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Input CSV must contain a '{TEXT_COLUMN}' column.")

    df[PREDICTION_COLUMN] = df[TEXT_COLUMN].apply(predict)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Score a holdout CSV of support-ticket messages."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV containing a text column.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path where predictions CSV should be written.",
    )

    args = parser.parse_args()

    score_holdout(args.input, args.output)


if __name__ == "__main__":
    main()