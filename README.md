# Yelp Review Rating Prediction

Predicting a review's star rating (1-5) from its text alone, using a subset of the Yelp Business Rating Prediction dataset (~10,000 reviews).
**Live demo:** [Live demo](https://yelp-review-api.onrender.com/) **API docs:** [API Docs](https://yelp-review-api.onrender.com/docs)

## Business framing

Review platforms and businesses want to understand sentiment at scale without reading every review by hand - flagging clusters of negative reviews early, routing them to support teams, or auto-flagging review sentiment for aggregate reporting. This project builds and compares three approaches to that problem, from a classical baseline to a fine-tuned transformer, and packages the best one as a live API.

## Why this project (vs. Project 1)

Project 1 (loan default prediction) was tabular, binary classification, deployed end-to-end. This project deliberately covers different ground: multi-class classification, text instead of tabular data, and a progression through three geniunely different NLP approaches - showing range rather than repeating the same skills on a new dataset.

## Project structure

```
yelp-review-rating/
├── README.md
├── data/
│   ├── raw/                 # original CSV, not committed — see data/raw/README.md
│   └── processed/            # cleaned train/val/test splits
├── notebooks/
│   ├── 01_eda.ipynb                              # class distribution, review length, word frequency
│   ├── 02_preprocessing_tfidf_baseline.ipynb      # text cleaning + TF-IDF + Logistic Regression/Naive Bayes
│   ├── 03_word_embeddings.ipynb                   # pretrained GloVe embeddings + classifier
│   ├── 04_transformer_finetune.ipynb              # fine-tuned DistilBERT
│   └── 05_model_comparison_error_analysis.ipynb   # compare all approaches, analyze failure modes
├── src/
│   ├── preprocessing.py     # shared text cleaning, reused across notebooks and app
│   ├── train_baseline.py    # trains + saves the TF-IDF baseline
│   └── predict.py           # loads the final model, runs inference
├── app/
│   ├── main.py       # FastAPI service exposing POST /predict
│   └── Dockerfile
├── models/            # saved model artifacts
├── tests/
│   └── test_preprocessing.py
├── requirements.txt       # full, for local dev (notebooks, tests)
├── requirements-app.txt   # lean, for deployment (mirrors Project 1's pattern)
└── .gitignore
```

## Status
-[X] EDA
-[X] Text preprocessing + TF-IDF baseline (Logistic Regression/Naive Bayes)
-[X] Word embeddings (GloVe) + classifier
-[X] Fine-tuned transformer (DistilBERT)
-[X] FastAPI service
-[X] Dockerized + deployed
-[X] Write-up

## Results
Two approaches were compared on a held-out test set (5000 reviews, never touched until the final evaluation):

| Model | Val Macro F1 | Test Macro F1 |
| --- | --- | --- |
| TF-IDF + Naive Bayes | 0.5531 | - |
| GloVe embeddings + Logistic Regression | 0.4756 | - |
| **TF-IDF + Logistic Regression (deployed)** | 0.5727 | **0.5687** |
| DistilBERT fine-tuned (full 14.4k training rows) | 0.6025 | **0.5829** |

Both leading models generalized well from validation to test (small, expected drops of ~0.4-2 points), indicating the modeling decisions made during development weren't overfit to the validation set despite rounds of experimentation.
**GloVe embeddings underperformed TF-IDF** - averaging word vectors discards word order entirely, argubaly losing more signal than even TF-IDF's simple bigrams. A legitimate, explainable negative result rather than a failed experiment.
**DistilBERT won on raw accuracy**, especially at the extremes (1,2, and 5-star reviews), consistent with a transformer's ability to use context and word order that TF-IDF's bag-of-words approach can't capture. Notably, DistilBERT matched TF-IDF's performance even when trained on **less than half the data** (6000 vs 14400 rows) before more training data widened the gap further - a clear denomstration of transfer learning' practical value.
**All approaches struggled most on 3-4 star reviews** - consistent across every model tested. Error analysis on DistilBERT's test-set mistakes showed this is driven largely by **mixed sentiment reviews** (e.g. geniune praise paired with a caveat), which are often geniunely ambiguous even to a human reader.

### Final model choice: TF-IDF, not DistilBERT
DistilBERT scored higher (0.5829 vs. 0.5687 macro-F1, ~2.5% relative improvement) but was **not** chosen for deployment. Reasoning:
* The accuracy gain is real but modest - not large enough to clearly justify the added deployment complexity for a portfolio-scale project
* TF-IDF's `joblib` artifact is ~1-5MB and needs only scikit-learn to serve; DistilBERT's fine-tuned weights are 250MB+ and require `torch`/`transformers` in the deployment image; a much heavier, slower-to-build, more failure prone deployment for this gain size
* This mirrors a real, common decision in applied ML: the most accurate model isn't always the one that ships. Being able to articulate *why* is arguably a stronger signal than defaulting to "use the best number"

DistilBERT's full training, evaluation, and error analysis are documented in `notebooks/04_transformer_finetune.ipynb` and `notebooks/05_model_comparison_error_analysis.ipynb` - evaluated throughly, deliberately not deployed. The fine-tuned weights themselves aren't committed to this repo (250MB+ exceeds GitHub's file size limit); re-run notebook 04 to regenerate them if needed.
## Limitations & next steps

### Data
* working set was a stratified ~18000-row subsample of the full ~650,000-row training set, and test evalation used a 5000-row subsample of the full 50,000-row test set - chosen for iteration speed ata causal development pace, not because the full dataset was infeasible. Results on the full dataset could differ, particularly for DistilBERT, which showed clear gains from more training data event within this reduced scale.
* Single domain (Yelp business reviews) - may not generalize to other review types (e.g. product reviews, which tend to be shorter and more feature focused).


### Modeling
* DistilBERT's `max_length=128` token truncation may be cutting off content in longer reviews (mean review length ~135 words); a longer max length might close some of the remaining gap on harder reviews, at the cost of slower training/inference.
* Mixed-sentiment reivews (praise + caveat) remain a hard case for every approach tested - geniunely ambiguous even for a human readers in some cases, but worth exploring whether a model explicitly designed for aspect-based sentiment (rating different aspects of an experience separately) would help.
* No hyperparameter search was run for any model; all four approaches used reasonable defaults or light manual tuning.

### Engineering
* Hit a real deployment constraint: GitHub rejects files over 100MB, which the fine-tuned DistilBERT weights exceed. Resolved by excluding the model directory from git and documenting how to regenerate it - a concrete, practical version of the accuracy-vs-deployment-cost tradeoff this project's final model choice is built around.

### Next steps
* Evaluate on the full 650k/50k dataset rather than subsamples
* Try a longer `max_length` for DistilBERT specifically
* If DistilBERT were to be deployed despite the cost trade-off, host the fine-tuned weights on Hugging Facee Hub and load via `from_pretrained()` at deploy time, rather than committed them to git.

## Setup
-`requirements.txt` - full set, for local development
-`requirements-app.txt` - lean set, for deployment

```bash
python -m venv venv
source venv/bin/activate #or venv/Scripts/activate on Windows
pip install -r requirements.txt
```

Download the data (see `data/raw/README.md`) before running notebooks.