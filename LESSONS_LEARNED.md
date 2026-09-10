# Lesson Learned - Yelp Review Rating Prediction (NLP Project)

## LinkedIN post 
I just wrapped up my second ML portfolio project: predicting 1-5 star rating from Yelp review text alone, comparing four approaches - TF-IDF, Naive Bayes, pretrained word embeddings, and a fine-tuned DistilBERT transformer.
The technical result: DistilBERT won on accuracy (0.58 vs. 0.57 macro-F1), and the part I actually find most interesting - it matched the TF-IDF baseline's performance using **less than half the training data**. That's transfer learning's whole value proposition in one number: a model pretrained on general language needed far less task-specific data to reach the same bar a classical model needed full data for.
But the decision I'm most proud of isn't the modeling - it's what I did *after* getting the result: I didn't deploy DistilBERT.
The accuracy gain was real but modest (~2.5% relative improvement). The deployment cost wasn't: 250MB+ model requiring GPU-friendly infrastructure, versus a 2MB scikit-learn pipeline that runs anywhere. For a project at this scale, that trade-off didn't favor a fancier model - so I shipped TF-IDF, and documented DistilBERT'S full evaluation in the repo as "evaluated, not deployed, here's why."
A few other real things I hit along the way:
* Downloaded the wrong dataset variant (binary sentiment instead of 5-star ratings) and only caught it because the class distribution looked suspiciously clean - a good reminder to verify data shape before trusting it
* Found a data cleaning bug, fixed it, then found a second, related bug hiding behind the first one - iterative debugging is normal, not a sign you did it wrong the first time
* Hit GitHub's 100MB file size limit trying to push the fine-tuned model - a very literal version of accuracy-vs-deployment-cost lesson above 
None of this happened in a single clean pass. All of it happened because I kept pushing past "model works in my notebook" toward "this is actually deployed and I can defend every decision I made"
[repo link](gitub.com/canary-jpg/yelp-review-prediction) | [live demo link](https://yelp-review-api.onrender.com/)
#MachineLearning #NLP #DataScience #MLOps #PortfolioProject

## Interview talking points

### Talking point 1 - Choosing the less accurate model (headline story)
**Situation:** After comparing four modeling approaches (TF-IDF, Naive Bayes, GloVe embeddings, fine-tuned DistilBERT) on a genuinely held-out test set, DistilBERT scored highest on macro-F1.
**Task:** I needed to decide which model to actually deploy
**Action:** I didn't default to "ship the highest-scoring model". I built out an explicit comparison of accuracy against deployment cost - model sizze, infrastructure dependencies, inference speed - and concluded the ~2.5% relative accuracy improvement didn't justify a 100x+ increase in deployment footprint (250MB+ and a GPU-friendly stack, vs a few MB and plain scikit-learn) for a project at this scale.
**Result:** Deployed the TF-IDF model, documented the full DistilBERT evaluation and reasoning in the repo rather than hiding it, so the decision-making process is visible, not just the final choice

### Talking point 2 - Catching a wrong dataset before it wasted a week
**Situation:** Early in the project, after downloading what I believe was a 5-class star-rating dataset, EDA showed the ratings were evenly split into only 2 values.
**Task:** I had to figure out whether this was a data processing bug on my end or something wrong with the source data itself.
**Action:** Rather than assuming my code was broken and debugging code that was actually fine, I stepped back and questioned the data itself - searched for what dataset variant would produce exactly that pattern (evenly split binary labels), and found I'd downloaded a binary sentiment dataset instead of the 5-class one the project actually needed. I switched sources and added an explicit verification step (checking label cardinality before proceeding) to the project's data documentation so the same mistake couldn't happen silently again.
**Result:** Caught in the first notebook, before any modeling time was wasted on the wrong problem framing.

### Talking point 3 - Iterative bug discovery in data cleaning
**Situation:** EDA revealed about half the review texts contained literal escaped newline characters. I wrote a cleaning fix and moved on.
**Task:** A later verification step (checking for orphaned single-letter tokens post-cleaning) surfaced a small number of cases that looked like genuine informal writing ("mac n cheess", "in n out") - but a few others looked like something else entirely: words split apart mid-word ("stapled i n a" instead of "stapled in a").
**Action:** Rather than assuming all remaining matches were the benign case, I looked at the actual matched text in context for each one, separated genuine informal usage from real artifacts, and traced the real artifacts to a related-but-different escape sequence (`\r`, not `\n`) that my first fix hadn't covered. Broadened the cleaning function to handle the whole family of escape sequences at once.
**Result:** A verification check that had returned "323 matches, mostly fine" became "3 matches, non concerning" after the second fix - geniunely clean data, confirmed rather than assumed. 

### Talking point 4 - Debugging a stale-object issue in a training pipeline
**Situation:** Fine-tuning a Hugging Face transformer, I hit a cryptic `ValueError: You must specify exactly one of input_ids or inputs_embeds` error, even after fixing what looked like the root cause (a label column naming mismatch).
**Task:** Determine why a fix that should have worked wasn't taking effect
**Action:** Instead of continuing to guess at new fixes, I added diagnostic print statements checking the dataset columns at every stage of the pipeline, confirmed the data itself was correct, and realized the actual `Trainer` object in memory was still pointing at stale dataset variables created *before* my fix - a notebook specific pitfall where variables persist across cell re-runs in ways that can mask whether a fix actually took effect.
**Result:** Rebuilding the `Trainer` object fresh, pointing explicitly at the corrected datasets, resolved it immediately.