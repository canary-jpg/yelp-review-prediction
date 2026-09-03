# Yelp Review Rating Prediction

Predicting a review's star rating (1-5) from its text alone, using a subset of the Yelp Business Rating Prediction dataset (~10,000 reviews).

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
-[] Text preprocessing + TF-IDF baseline (Logistic Regression/Naive Bayes)
-[] Word embeddings (GloVe) + classifier
-[] Fine-tuned transformer (DistilBERT)
-[] FastAPI service
-[] Dockerized + deployed
-[] Write-up

## Results

## Limitations & next steps

## Setup
-`requirements.txt` - full set, for local development
-`requirements-app.txt` - lean set, for deployment

```bash
python -m venv venv
source venv/bin/activate #or venv/Scripts/activate on Windows
pip install -r requirements.txt
```

Download the data (see `data/raw/README.md`) before running notebooks.