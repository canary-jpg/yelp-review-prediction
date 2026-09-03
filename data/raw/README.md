# Raw data

Not committed to the repo (see `.gitignore`).

Using the **Yelp Review Full** dataset (Zhang et al., 2015) — the 5-class
("fine-grained") benchmark version, ~650,000 train / 50,000 test reviews:

https://www.kaggle.com/datasets/yacharki/yelp-reviews-for-sa-finegrained-5-classes-csv

**Easy mix-up to avoid:** this dataset is commonly confused with **Yelp
Review Polarity** (binary — labels 1/2 only, 3-star reviews dropped, ~560k
rows, exactly 50/50 balanced), which shares the same paper and often sits
on similar-looking Kaggle pages. Verify what you actually downloaded
**before** running any notebook:

```python
import pandas as pd
df = pd.read_csv("data/raw/train.csv", header=None, names=["stars", "text"])
print(df["stars"].value_counts().sort_index())
```

You should see **5 distinct values (1-5)**, each roughly 100k-140k rows —
not just two values at ~280k each (that would mean you have Polarity, not
Full).

**File format:** `train.csv` and `test.csv`, no header row, two columns:
1. `label` — star rating, 1-5
2. `text` — review body, quoted

Place both files at `data/raw/train.csv` and `data/raw/test.csv`.

**Why a subsample:** 650k rows is more than needed for a casual-pace
portfolio project, and would make transformer fine-tuning (notebook 04)
slow to iterate on. `01_eda.ipynb` takes a stratified sample of `train.csv`
(~15,000-20,000 rows) as the working dataset for EDA through model
development.

**`test.csv` stays untouched** until `05_model_comparison_error_analysis.ipynb`
— using it only once, at the very end, gives this project a genuine
held-out test set, which Project 1 (loan default prediction) didn't have.
Don't peek at it or tune anything against it before then, or it stops being
a valid final check.
