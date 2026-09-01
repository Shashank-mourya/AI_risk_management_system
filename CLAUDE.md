# AI Risk Manager — Return Risk Scorer

Razorpay hackathon, **Track 02 — AI Risk Manager**.

Given an order at the moment it is placed, output the probability it will be
returned, so the merchant can act before absorbing the loss.

**The graded bar is "honest metrics including false-positive cost."** Evaluation
rigour outranks model accuracy on every trade-off in this repo. If a change makes
the model better but the metrics less trustworthy, it is the wrong change.

---

## Decisions already made (do not relitigate without a reason)

| Decision | Why |
|---|---|
| **Return-risk, not fraud** | Fraud is a different track. Fraud = someone else's card. Return risk = the real cardholder, transaction still unwinds. |
| **Returns only, chargebacks unevaluated** | No public dataset carries disputes. `label_disputed` is always 0. The dispute path is built in the schema and API but **no metrics are reported for it** — synthetic dispute labels would measure our own generator, which fails the honesty bar. Say this out loud in the write-up. |
| **Dataset: UCI Online Retail II** | 1,067,371 rows, Dec 2009 – Dec 2011. Real returns via credit notes (`Invoice` starting with `C`). Real customer IDs, minute-level timestamps. Razorpay supplied nothing. |
| **Amounts stay in GBP** | UK retailer. Converting to INR would imply the data is Indian when it is not, and there is no honest 2011→2026 rate. Provenance beats familiarity. |
| **Two candidate models, ship one** | Logistic regression (interpretable baseline) + LightGBM. Same features, same rows, same split. Selected on **total cost at the optimal threshold** — not accuracy, not F1. If boosting wins only marginally, ship the simpler model and say why. |
| **Streamlit, not a React frontend** | The differentiator is the cost-vs-threshold curve, which needs interactive cost sliders. FastAPI only if time remains. |
| **One LLM, stateless** | `openai/gpt-oss-120b` on Groq (`GROQ_MODEL` in `.env`). One call, score + top features in, one paragraph out. **Not a chatbot** — no history, no follow-ups, no routing by severity. Was `llama-3.3-70b-versatile`; Groq retired the Llama chat models and the key 404s on them. The model name is a setting, not a decision — the invariance test exists precisely so swapping it changes nothing that matters. |

## Hard rules

1. **No leakage.** Every feature for an order at time `T` uses only events that had
   *already happened* by `T` — measured by when the outcome occurred, not by an
   earlier order's eventual label. This is the single most important invariant here.
2. **Chronological splits only.** Never `train_test_split`. Never plain k-fold. Use
   `TimeSeriesSplit` on training rows if cross-validating.
3. **No resampling.** 18% positive needs no SMOTE, and rebalancing would distort the
   calibrated probabilities the cost model depends on.
4. **The LLM never decides.** It writes prose from a finished score. There is no code
   path where generated text can alter `risk_probability`, `risk_band` or
   `recommendation`. Keep it that way — it is a graded deliverable.
5. **Defence only.** No blocklisting, no cross-account identity graphs, no scoring a
   person rather than a transaction. Offence-capable work is disqualified.
6. **Human in the loop.** A score produces a *recommendation*. Any action requires a
   named `reviewer_id`.

## Measured numbers (do not guess these)

```
rows                    1,067,371      Dec 2009 - Dec 2011
source sha256      be2480b1fcb1fa12    pinned in config.py, verified on load
no CustomerID              22.8%       excluded: returns unattributable
credit-note lines          18,744      89.0% matched to a purchase
same-day matches            11.1%      clerical corrections, dropped
beyond the 90d window        1,572     outside the label horizon, dropped
non-credit negatives        3,457      'damages'/'check' - merchant write-offs, excluded

gap p50 / p90 / p95     10d / 84d / 150d
MIN_GAP_DAYS                  1.0      below this it is a clerical fix, not a return
RETURN_WINDOW_DAYS             90      the LABEL horizon AND the maturity horizon

mature orders              30,347      6,628 immature dropped (inside 90d of data end)
POSITIVE RATE              16.73%
split date             2011-04-28      train 24,277 (16.83%) / test 6,070 (16.33%)
order value GBP      median 304.44     mean 470.80
```

**The label is "returned within [1, 90] days", not "ever returned".** It used to be
the latter, which meant an order from Dec 2009 was watched for 667 days and one from
Sep 2011 for 90 - and the positive rate rose with the length of the watch (17.6% in
the shortest-window quintile against 20.3% in the longest). That is exposure time, not
risk. Because the split is chronological, train sat in the long-window end and test in
the short one, manufacturing a 1.21pp train/test gap out of nothing; capping the label
at the window maturity already guarantees cut it to 0.50pp. `test_phase1.py` check 4
now fails if the positive rate drifts across observation-window quintiles.

## Pipeline

```
config.py           → the ONE label definition + every shared constant
build_labels.py     → retail2.pkl, orders_labeled.pkl/.csv    (label + maturity + split)
build_features.py   → features.pkl/.csv                       (17 as-of features)
build_database.py   → risk.db                                 (30 tables, 9 loaded)
cost_model.py       → the ONE cost model, imported never redefined
train_model.ipynb   → model, threshold sweep, cost curve      (CPU — GPU is pointless at 24k rows)
predict.py          → the ONE scoring path                    (schema vocabulary)
score_to_db.py      → risk_scores + risk_score_features       (joins phase 2 to the db)
explain.py          → risk_explanations                       (one Groq call, stateless)
app.py              → demo surface                            (streamlit run app.py)
```

**Four times now, the same failure:** a thing that should exist once existed twice, and
the two copies disagreed. The cost model (two "optimal" thresholds). The scoring path
(a copy in a test file emitting recommendations the schema forbids). The label window
(defined in six files, one under the comment "must match build_labels.py"). The band
boundary (`min(thr*2, 1.0)` in the database vs an unclipped `thr*2` in the scorer).
Before adding a constant or a formula, check whether it already lives somewhere.

`.pkl` for training (dtypes survive), `.csv` for inspecting. Both are regenerable —
keep them out of git.

**Feature note:** `customer_prior_return_rate` uses `-1` as a sentinel for customers
with no history, flagged by `is_new_customer`. Do not impute it to 0 — "no history"
and "never returned" are different states. 17.4% of orders are cold-start.

`customer_prior_returns` counts prior **orders** observed returned, not prior returned
*line items*. It once counted lines against an order denominator, which made the "rate"
reach 45.0. `build_features.py` now asserts `prior_returns <= prior_orders`, and
`test_phase1.py` check 8 re-derives the whole thing independently.

`basket_sku_return_rate`, `basket_max_sku_return_rate` and `price_vs_sku_mean` are all
built **as of the split date** — only purchases made before it and only returns
*observed* before it. The SKU rate previously used train-period purchases with their
*eventual* outcomes, 7.3% of which resolved after the split; `price_vs_sku_mean`
averaged over the whole dataset, pricing 474 SKUs entirely from rows that did not exist
yet. Both went stale-but-honest instead, which is what a deployed model actually has.

## Reference documents

- `AI_Risk_Manager_API_Reference.pdf` — 65 endpoints. **A design artifact, not a build
  list.** Implement ~3: `POST /risk/score`, the explanation endpoint, `/health`.
- `AI_Risk_Manager_DB_Schema_FINAL.pdf` — 30 tables, the v1→v3 changelog, dataset binding.
- `AI_Risk_Manager_schema_v3.sql` — runnable DDL, SQLite + Postgres, dependency-ordered.

Razorpay fidelity details that were verified against live docs and are easy to get
wrong again: `created_at` is an **integer Unix epoch**, not a timestamp. Dispute
`phase` has five values including `fraud` and `retrieval` — both are **out of scope
and must be excluded from the label**. Payment `method` is card/netbanking/wallet/emi/upi;
`cod` is our own extension and is labelled as such.

## Model (Phase 2, settled)

Logistic regression ships. LightGBM was 0.2% cheaper — below the pre-committed 2% bar,
so the simpler and directly interpretable model wins.

```
threshold              0.17       cost-optimal, not F1-optimal, not 0.5
analytic break-even   0.198       cost model alone, no data - close agreement
precision             0.330       2.02x the 16.33% base rate
recall                0.614
ROC-AUC               0.748       PR-AUC 0.375 (2.30x base rate)
accuracy              0.734       flag-nothing floor is 0.837 - quote both
ECE                   0.017       calibration is what makes the cost curve mean anything
Brier                 0.121
confusion       TP 608  FP 1232  FN 383  TN 3847
flag rate             30.3%
```

**The cost model lives in `cost_model.py` and nowhere else.**

```
total = (tp+fp)*review + fp*friction + fn*cost_return + tp*cost_return*(1-prevention)
```

Both trailing terms were missing from the original `fp*c_fp + fn*c_fn`: reviewing a
flagged order costs money whether or not the flag was right, and catching a return is
not the same as preventing one. Correcting them moved the optimum from 0.07 (flagging
81% of all orders — the cost ratio talking, not the model) to 0.17.

Two representative order values, not one: a miss is priced on what returned orders are
worth (median £388.15), a false alarm on what kept orders are worth (median £285.59).

**Measured:** those two order values. **Assumed:** recovery rate, PSP fee, logistics,
review cost, abandon rate, margin, prevention rate — this dataset carries no cost data.
Report the *shape* (optimum near 0.2, well below 0.5, moves slowly), not the pounds.

## Scoring and explanation (Phase 3)

`predict.py` is the only thing that produces `risk_probability`, `risk_band` and
`recommendation`. It was consolidated from two divergent inline copies, one of which
sat in a test file under a comment calling itself "the single scoring path". That copy
emitted `approve`/`review`/`hold_for_review` — **none of which are legal** under the
schema's CHECK constraint. Nothing caught it because nothing had ever written a score
to the database. Use the schema vocabulary: bands are `low`/`medium`/`high`,
recommendations are `allow`/`manual_review`/`hold_payout`/`request_verification`.

Bands sit at the threshold and at twice the threshold, so they move when it does.

`explain.py` only ever SELECTs from `risk_scores` and INSERTs into `risk_explanations`.
`test_explain.py` check 3 feeds it a model whose every reply demands
`risk_probability=0.0, recommendation=allow`, and requires the decision back
bit-identical. Keep that test passing — it is the AI/non-AI deliverable.

## Still open

- Write-up / submission package.
- A `GROQ_API_KEY` in `.env` — everything in Phase 3 is built and tested against
  stubs, but no real paragraph has been generated yet.

## Known limitations (state these, do not hide them)

UK **wholesale gift retailer** — median order £304, customers are mostly businesses,
so return behaviour is B2B-flavoured rather than consumer. 22.8% of rows have no
customer ID and are excluded entirely. No payment method, no addresses, no discount
data in the source, so those planned features do not exist.
