"""
Shared text preprocessing, used by notebooks 02-04 and by the deployed API 
(src/predict.py) - keeping this in one place means training-time and 
inference-time cleaning can never silently drift apart, same principle as 
Project 1's pipeline.py
"""

import re 

def clean_text(text:str) -> str:
    """baseline cleaning for the TF-IDF approach (notebook 02)
    Kept deliberating simple: lowercase, strip non-alphanumeric noise,
    collapse whitespace. Word embedding and transformer approaches (notebook 03-04) may need
    a lighter touch (e.g. transformers generally do better with more of the original text,
    including puntuation, left intact) - revisit this function's use per-notebook
    rather than assuming one cleaning strategy fits every model type. """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.replace("\\n", " ")
    text = text.replace("\n", " ") #also handles genuine newlines, just in case
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text) #non-alphanumeric
    text = re.sub(r"\s+", " ", text).strip() #collapsing whitespace

    return text