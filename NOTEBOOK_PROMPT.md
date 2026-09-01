> **Note.** This is the prompt that generated `notebooks/train_model.ipynb`.
> The notebook has since been edited directly — it imports `cost_model.py`
> rather than defining a cost model inline. Treat this file as a record of
> how the notebook started, not as a spec for regenerating it.

# Prompt — training notebook

Paste everything below the line into Opus with high reasoning effort.

---

You are writing a single Jupyter notebook for a hackathon submission. It runs locally on CPU. Write the complete notebook as ordered cells (markdown + code), ready to paste in. Do not explain your plan first — produce the notebook.

## Context

Razorpay hackathon, Track 02 "AI Risk Manager". We are building a **return-risk scorer**: given an order at the time it is placed, output the probability it will be returned, so the merchant can act before absorbing the loss.

The graded bar is: **"Honest metrics including false-positive cost."** Measured precision and recall on a held-out test set. The evaluation rigour matters more than model accuracy. Judges will probe whether the metrics are trustworthy, so every number the notebook prints must be defensible.

## Data

One file, already feature-engineered and split. Load with:

```python
df = pd.read_pickle("features.pkl")   # repo root
```

30,347 rows. Columns:

**Identifiers / metadata (never features):** `Invoice`, `customer_id`, `order_date`, `split`
**Label:** `returned` (int 0/1)
**Split column:** `split` is `"train"` (24,277 rows, 16.83% positive) or `"test"` (6,070 rows, 16.33% positive)

**The 17 features:**
`order_value`, `log_order_value`, `n_lines`, `total_quantity`, `mean_unit_price`, `max_unit_price`, `price_vs_sku_mean`, `hour_of_day`, `day_of_week`, `is_uk`, `customer_prior_orders`, `customer_prior_returns`, `customer_prior_return_rate`, `customer_tenure_days`, `is_new_customer`, `basket_sku_return_rate`, `basket_max_sku_return_rate`

Notes that affect how you handle them:
- `customer_prior_return_rate` uses **-1 as a sentinel** for customers with no history. Do not impute it; `is_new_customer` flags those rows. Trees handle the sentinel natively. For logistic regression, keep the sentinel but rely on `is_new_customer` to let the model separate the two regimes.
- All features are already computed **as-of the order timestamp** — no future information. Do not engineer any new feature that uses the full dataset, and do not refit any encoder on train+test combined.
- The split is **chronological, not random.** Never reshuffle. Never use `train_test_split`. Never use plain k-fold cross-validation — if you cross-validate for hyperparameters, use `TimeSeriesSplit` on the training rows only, ordered by `order_date`.

Amounts are in **GBP** (source data is a UK retailer). Keep them in GBP; do not convert.

## What the notebook must do

**1. Setup and load.** Import, load, print shape and class balance per split. Assert the split is chronological: `train.order_date.max() <= test.order_date.min()`.

**2. Quick EDA.** Positive rate over time (monthly), feature distributions by class, correlation of each feature with the label on train only. Keep it brief — three or four plots.

**3. Train two candidate models** on the same features, same rows, same target:
   - **Logistic Regression** — the interpretable baseline. Standardise features (fit the scaler on train only).
   - **LightGBM** — gradient boosting. Modest hyperparameters; this dataset is small and will overfit if you let it.

   Use **CPU only.** 24k rows × 17 features trains in well under a second; GPU is pointless at this size and adds transfer overhead. Do not set any GPU parameters.

**4. Evaluate both on the held-out test set.** Report for each: precision, recall, F1, ROC-AUC, PR-AUC, and the confusion matrix. Report the **base rate (16.33%)** alongside, and state explicitly what a trivial always-predict-positive baseline would score, so the reader can see what the model is actually adding. Plot the precision-recall curve for both on the same axes.

**5. The cost model — this is the centrepiece.** Define, with the arithmetic visible in code and stated as formula inputs, not bare constants:

```
cost_per_fp = lost_sale_probability * average_order_value + trust_penalty
cost_per_fn = refunded_amount + return_shipping + restocking_cost
```

Use the dataset's own average order value (mean ≈ £470, median ≈ £304) to ground these, and make every assumption an explicitly named variable at the top of the cell so a reader can change one and re-run. Explain in a markdown cell why the two costs are not symmetric.

**6. Threshold sweep.** For each model, sweep thresholds from 0.01 to 0.99 in steps of 0.01. At each threshold compute TP/FP/FN/TN, precision, recall, and `total_cost = fp * cost_per_fp + fn * cost_per_fn`. Produce:
   - a **cost-vs-threshold plot** with the minimum marked and annotated
   - a table of metrics at the cost-optimal threshold, the default 0.5, and the F1-optimal threshold, side by side

   Make the point explicitly: the cost-optimal threshold is **not** 0.5, and is not the F1-optimal threshold either.

**7. Sensitivity analysis.** Vary the FP/FN cost ratio across a range (say 1:1 up to 1:10) and plot how the optimal threshold moves. This demonstrates that the threshold is a business decision, not a model constant.

**8. Model selection.** Choose the winner on **total cost at its own optimal threshold**, not on accuracy or F1. If LightGBM beats logistic regression only marginally, say so and argue for shipping the simpler model. Make the selection criterion explicit.

**9. Interpretability.** Logistic regression coefficients (standardised, sorted) and LightGBM feature importance (gain). Compare them — where do the two models agree on what matters? Note whether the ranking is consistent with the correlations from step 2.

**10. Save artefacts** to the repo root: the winning model via joblib, the scaler if used, a JSON with the chosen threshold and the cost-model inputs, and the threshold sweep as CSV.

## Style requirements

- Every non-obvious choice gets a **markdown cell explaining why**, written for a judge, not for a data scientist.
- No `train_test_split`, no `SMOTE`, no resampling. An 18% positive rate does not need rebalancing, and rebalancing would distort the calibrated probabilities the cost model depends on. Say this in a markdown cell rather than silently not doing it.
- Set `random_state=42` everywhere applicable.
- Plots: matplotlib only, clearly labelled axes, titles, no seaborn styling dependencies.
- The notebook must run top to bottom without errors from a clean kernel.
- Do not print more than ~20 rows of any dataframe.

## What good looks like

The reader should finish the notebook able to answer: how well does this model work, how do I know the number is honest, what does it cost the business to be wrong in each direction, and what threshold should we actually ship — with the evidence for each visible on screen.
