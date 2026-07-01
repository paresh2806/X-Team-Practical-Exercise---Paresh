import argparse

from ticket_classification.predict import predict
from ticket_classification.score_holdout import score_holdout


def main():
    parser = argparse.ArgumentParser(
        description="Support ticket classification CLI"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict label for a single support-ticket message"
    )
    predict_parser.add_argument(
        "text",
        help="Support-ticket message text"
    )

    score_parser = subparsers.add_parser(
        "score",
        help="Score a CSV file and write predictions"
    )
    score_parser.add_argument(
        "--input",
        required=True,
        help="Input CSV path. Must contain a text column."
    )
    score_parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path for predictions."
    )

    args = parser.parse_args()

    if args.command == "predict":
        prediction = predict(args.text)
        print(prediction)

    elif args.command == "score":
        score_holdout(args.input, args.output)
        print(f"Predictions written to: {args.output}")


if __name__ == "__main__":
    main()