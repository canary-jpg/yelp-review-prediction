"""
Trains the TF-IDF + Logistic Regression baseline and saves it.
Usage:
    python src/train_baseline.py --data data/processed/train.csv
"""

import argparse 
import joblib 
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import Pipeline 

from preprocessing import clean_text 

TARGET = "stars"
TEXT_COL = "text"

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TEXT_COL] = df[TEXT_COL].apply(clean_text)
    return df 

def main(data_path: str, model_out: str):
    df = load_data(data_path)

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
            (
                "model",
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", multi_class="multinomial"
                ),
            ),
        ]
    )

    pipeline.fit(df[TEXT_COL], df[TARGET])
    #TODO: evaluate on a held out val set (macro-F1, confusion matrix)
    #rather than just fitting and saving (see notebook 02 for full evaluation)
    #this script is meant for quick retraining, not analysis

    joblib.dump(pipeline, model_out)
    print(f"Saved TF-IDF baseline pipeline to {model_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/train.csv")
    parser.add_argument("--model-out", default="models/tfidf_baseline.joblib")
    args = parser.parse_args()

    main(args.data, args.model_out)