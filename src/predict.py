"""
Loads the final trained model and runs predictions on new review text.
Shared by app/main.py so training and serving never drift apart.

FINAL MODEL CHOICE: TF-IDF + Logistic Regression (models/tfidf_baseline.joblib).

Notebook 05 found DistilBERT (fine-tuned on the full training set) scored
higher on the held-out test set (0.5829 vs. 0.5687 macro-F1) - a real but
modest ~2.5% relative improvement. TF-IDF was chosen for deployment anyway: the 
accuracy gain didn't justify DistilBERT's much higher deployment cost (~250MB+ artifact
requiring torch/transformers vs. a ~1-5MB scikit-learn pipeline). The DistilBERT model, its
full evaluation and this reasoning are documented in notebooks 04-05 - evaluated
throughly, deliberately not deployed. See README for full write up
"""

import joblib 
from preprocessing import clean_text 

MODEL_PATH = "models/tfidf_baseline.joblib"

_MODEL = None 

def get_model(model_path: str = MODEL_PATH):
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(model_path)
    return _MODEL 

def predict_rating(review_text: str, model_path: str = MODEL_PATH) -> dict:
    model = get_model(model_path)
    cleaned = clean_text(review_text)
    predicted_class = int(model.predict([cleaned])[0])
    probabilities = model.predict_proba([cleaned])[0]
    class_probs = {
        str(cls): float(prob) for cls, prob in zip(model.classes_, probabilities)
    }
    return {'predicted_stars': predicted_class, 'class_probabilities': class_probs}
    
