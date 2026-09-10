"""
FastAPI service exposing the review rating model
Run locally:
    uvicorn app.main:app --reload --app-dir
Then POST to /predict, e.g.:
    curl -X POST http://localhost:8000/predict -H "Content-Type: application.json" \
        -d '{"review_text": "The food was amazing and the service was great!"}'
"""

import sys 
from pathlib import Path 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from predict import predict_rating #noqa: E402

app = FastAPI(
    title="Yelp Review Rating Prediction API",
    description="Predicts a 1-5 star rating from review text alone.",
)

class ReviewInput(BaseModel):
    review_text: str 

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(review: ReviewInput):
    try:
        return predict_rating(review.review_text)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model artifact not found - train the model first (see notebooks).",
        )