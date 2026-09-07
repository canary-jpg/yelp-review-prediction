"""
Tiny shared results tracker so the model comparison in notebook 05 doesn't
require manually re-copying metrics from eariler notebooks. Each notebook
calls save_result() after evaluating its model; notebook 05 calls
load_results() to build the final comparison table
"""

import json 
from pathlib import Path 

RESULTS_PATH = Path(__file__).resolve().parent.parent/ "data" / "processed" / "model_comparison.json"

def save_result(model_name: str, metrics: dict, path=RESULTS_PATH) -> dict:
    path = Path(path)
    if path.exists():
        with open(path) as f:
            results = json.load(f)
    else:
        results = {}
    results[model_name] = metrics 
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return results 

def load_results(path=RESULTS_PATH) -> dict:
    path = Path(path)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}