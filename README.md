# Support Ticket Classification

Classical machine learning solution for routing customer support tickets into one of four labels:

- `account-access`
- `transaction-dispute`
- `fraud-report`
- `general`

The final model is a TF-IDF + Logistic Regression pipeline with `class_weight="balanced"`, saved at:

```text
models/v2_tfidf_logreg_balanced.joblib
```

## Project Structure

```text
data/train.csv                     Training data
notebooks/                         EDA and model experiments
models/                            Saved trained models
ticket_classification/             Prediction, scoring, and CLI code
sample_data/sample_holdout_20.csv  Sample holdout input
tests/                             Unit and CLI tests
```

## Approach

I used `notebooks/EDA.ipynb` first to understand the data before modeling. The notebook showed that the dataset has 400 rows, two useful columns (`text` and `label`), no missing values, and no duplicate text rows. It also showed an imbalanced label distribution, with `fraud-report` as the smallest class.

That shaped the modeling approach: I treated this as a supervised multiclass text classification problem, used a stratified validation split, tracked macro F1 instead of only accuracy, and paid extra attention to fraud recall. The EDA also showed meaningful class-specific words and phrases, which made TF-IDF a good fit for a simple first model.

Since classical ML models cannot train directly on raw text, the ticket messages needed to be converted into numeric features first. I chose TF-IDF because it captures which words are important within a message while reducing the impact of very common words. For this small dataset, TF-IDF was also easier to explain, faster to train, and more appropriate than adding heavier text representation methods.

I compared three model versions:

| Version | Model | Result |
|---|---|---|
| V1 | TF-IDF unigrams + Logistic Regression | Strong baseline, but missed one fraud example |
| V2 | TF-IDF unigrams + balanced Logistic Regression | Best selected model |
| V3 | TF-IDF unigrams + bigrams + balanced Logistic Regression | Matched V2, but added complexity |

V2 was selected because it matched the best validation performance while staying simpler than V3.

## Validation Metrics

Validation used an 80/20 stratified split.

| Model | Accuracy | Macro F1 | Weighted F1 | Fraud Recall |
|---|---:|---:|---:|---:|
| V1 | 0.9875 | 0.9830 | 0.9873 | 0.9000 |
| V2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| V3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Macro F1 was used as the main metric because the classes are imbalanced. Fraud recall was tracked separately because missing fraud-related tickets is higher risk.

## Setup

Requires Python 3.13+ and `uv`.

```bash
uv sync
```

## Single Prediction

```bash
uv run python -m ticket_classification.cli predict "I cannot login to my account"
```

Example output:

```text
account-access
```

## Holdout Scoring

Input CSV must contain a `text` column.

```bash
uv run python -m ticket_classification.cli score \
  --input sample_data/sample_holdout_20.csv \
  --output sample_data/sample_predictions_20.csv
```

The output CSV includes a new `predicted_label` column.

## Tests

```bash
PYTHONPATH=. uv run pytest
```

Current result:

```text
11 passed
```

## What I Prioritized

I focused on getting the core workflow right: understanding the data, building a simple baseline, comparing a few model choices, saving the final model, and making it easy to run predictions from code or the CLI.

I kept the project intentionally small. I did not add:

- LLM-based classifier comparison
- FastAPI service
- Docker/containerization
- CI pipeline

Those would be useful in a larger production setting, but for this assignment they would mostly add surface area. The selected model is deterministic, fast, explainable, and easy to test, which felt like the better trade-off for a small dataset.

With more time, I would add cross-validation, confidence scores, deeper error analysis on hidden holdout results, and more classical baselines such as CountVectorizer or TF-IDF with Naive Bayes. I would also consider a small FastAPI wrapper, and possibly an LLM as a comparison or fallback for ambiguous tickets, not as the first choice for this dataset.

Approximate time spent: **3-3.5 hours**.
