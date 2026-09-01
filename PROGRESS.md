# AI Risk Manager — Progress Summary

> **Last updated:** 1 Sep 2026  
> **Goal:** Return-risk scorer — given an order at placement time, output the probability it will be returned.  
> **The bar:** *"Honest metrics, including false-positive cost."*  
> **Public face:** `README.md` — de-branded, no hackathon or sponsor framing.  

---

## What's Done ✅

### Phase 1 — Data Pipeline (fully complete)

Three scripts, run in order. All outputs are regenerable from source.

| Script | What it does | Output |
|---|---|---|
| `build_labels.py` | Downloads UCI Online Retail II (~1M rows), matches credit notes to purchases, applies 1-day clerical filter + 90-day maturity cutoff, writes chronological train/test split. | `retail2.pkl`, `orders_labeled.pkl/.csv` |
| `build_features.py` | Computes 17 as-of features (no future information). Customer history counted by *when the return happened*, not by an earlier order's label. | `features.pkl/.csv` |
| `build_database.py` | Creates all 30 tables from `AI_Risk_Manager_schema_v3.sql` (SQLite), loads 9 of them. | `risk.db` (~139 MB) |

**Verified numbers:**
```
Source sha256:           be2480b1fcb1fa12...  pinned in config.py, verified on load
Label:                   returned within [1, 90] days
Mature orders:           30,347   (6,628 immature dropped)
Positive rate:           16.73%
Train / Test:            24,277 (16.83%) / 6,070 (16.33%)
Split date:              2011-04-28  (chronological, NOT random)
FK violations:           0
```

### Phase 2 — Model Training (fully complete)

- **Notebook:** `notebooks/train_model.ipynb` — runs top-to-bottom on CPU
- **Cost model:** `cost_model.py` — the single definition, imported by the notebook,
  `evaluate_model.py` and (later) the app. It previously existed in two places that
  disagreed, which put two different "optimal" thresholds in the repo at once.
- **Two models trained:** Logistic Regression (with StandardScaler) and LightGBM
- **Winner:** Logistic Regression — chosen on **total cost**, not accuracy
  - LightGBM came out **0.2% more expensive** at its own cost-optimal threshold, so
    there was no simplicity-versus-accuracy trade-off to argue about
  - The `MARGINAL_GAIN_THRESHOLD` rule was pre-committed and is recorded in
    `threshold.json`, so it is visibly not chosen after seeing the result

**Model performance (held-out test set, at cost-optimal threshold 0.17):**
```
Precision:    0.330      2.02x the 16.33% base rate
Recall:       0.614
F1:           0.430
Accuracy:     0.734      flag-nothing floor is 0.837 — quote both or neither
ROC-AUC:      0.748
PR-AUC:       0.375      2.30x base rate
ECE:          0.017      calibration is what makes the cost curve mean anything
Brier:        0.121
Flag rate:    30.3%
```

**Confusion matrix:**
```
TP:  608    FP: 1232
FN:  383    TN: 3847
```

**Cost model (GBP pence)** — `cost_model.py`:
```
total = (tp+fp)*review + fp*friction + fn*cost_return + tp*cost_return*(1-prevention)

MEASURED on train rows
  median value, returned orders   38,815   (GBP 388.15)   prices a MISS
  median value, kept orders       28,559   (GBP 285.59)   prices a FALSE ALARM
DERIVED
  cost of a return                15,212   (GBP 152.12)
  cost of a false alarm              503   (GBP   5.03)
  review, charged on EVERY flag      500
  analytic break-even p*          0.1980   vs empirical optimum 0.17
```

Two corrections over the naive `fp*c_fp + fn*c_fn` that the earlier version used:
reviewing a flagged order costs money whether or not the flag was right, and catching
a return is not the same as preventing one (assumed 30% prevention, so a true positive
still costs 70% of a miss). The naive form bottomed out at threshold **0.07, flagging
81% of all orders** — the cost ratio talking, not the model. The notebook's own
degeneracy guard caught that and said so. The corrected form lands at **0.17**.

> **What is and is not defended.** The two order values are measured. The other seven
> inputs — recovery rate, PSP fee, logistics, review cost, abandon rate, margin,
> prevention rate — are industry assumptions, because this dataset carries no cost data
> at all. Report the *shape* (optimum near 0.2, well below the default 0.5, and it moves
> slowly), not the pound figures. `threshold_sensitivity.png` sweeps the two that
> actually move the answer.

**Saved artefacts** (in `artefacts/`):
| File | What |
|---|---|
| `model.joblib` | Trained logistic regression model |
| `scaler.joblib` | StandardScaler (fit on train only) |
| `threshold.json` | Chosen threshold, full cost-model inputs, holdout metrics, cost baselines, feature list |
| `threshold_sweep.csv` | Threshold sweep 0.01→0.99 with TP/FP/FN/TN/cost at each point |
| `cost_vs_threshold.png` | Cost-vs-threshold curve, minimum and analytic break-even both marked |
| `threshold_sensitivity.png` | Two panels: sensitivity to cost-of-a-return and to prevention rate |

**Fixed during Phase 1 acceptance:** `customer_prior_returns` counted returned *line
items* against a denominator of *orders*, so `customer_prior_return_rate` was not a rate
— it reached 45.0 on 2,378 rows. It now counts prior orders observed returned, dated at
the earliest line that came back. `build_features.py` asserts the bound and
`test_phase1.py` check 8 re-derives it independently. Fixing it also lifted ROC-AUC at the time.

### Supporting Scripts (complete)

| Script | Purpose |
|---|---|
| `make_synthetic_dataset.py` | Generates a synthetic stand-in for `features.pkl` — used for Kaggle uploads and CI/smoke tests where the real data isn't available. Emits the same 17-column schema. **Not for evaluation** — any metrics on synthetic data measure the generator, not the model. |
| `test_phase1.py` | 72-check acceptance suite for the data pipeline. Check 8 re-derives the as-of customer history with a slow explicit loop and demands an exact match against the vectorised implementation — two independent implementations agreeing is the only real evidence hard rule #1 holds. |
| `cost_model.py` | The one cost model. Imported by the notebook, `evaluate_model.py` and the app; never redefined. Separates measured inputs from assumed ones. |
| `test_model.py` | 26-check invariant harness for saved artefacts. Wiring, determinism, sentinel handling, threshold contract, band monotonicity — not accuracy. Works on both real and synthetic data. |
| `evaluate_model.py` | Part A: 27 edge-case checks (pass/fail). Part B: held-out accuracy with bootstrap CIs, calibration deciles, lift, segment breakdown, temporal stability (measurement, no pass/fail). |

### Integration — Phase 1 ↔ Phase 2 (complete)

The two phases were each internally correct but **not joined**. `models`,
`threshold_config`, `risk_scores` and `risk_score_features` were all empty, so the
scoring half of a 30-table schema was decoration — and `risk_explanations.score_id` is
a foreign key onto `risk_scores(id)`, so Phase 3 had nothing to key a cache on.

| File | What it fixes |
|---|---|
| `predict.py` | The one scoring path. Was duplicated inline in `test_model.py` and `evaluate_model.py`; the copy that called itself "the single scoring path" lived in a test file nothing imports. |
| `score_to_db.py` | Writes the model, the threshold config, 6,070 scores and 103,190 feature snapshots into `risk.db`. Deterministic ids, idempotent re-runs. |

**A vocabulary bug this surfaced:** the inline scorer emitted recommendations
`approve` / `review` / `hold_for_review`. The schema CHECK allows only
`allow` / `manual_review` / `hold_payout` / `request_verification`. Every score would
have been rejected by the database. Nothing caught it because nothing had ever written
one. `predict.py` now uses the schema's vocabulary and `test_phase1.py` asserts it.

**The connection is proved, not assumed** — `score_to_db.py` recomputes recall in SQL
across `risk_scores × risk_labels × dataset_members` and requires it to match
`threshold.json`:

```
scores joined to labels via payment_id   6,070
actual returns among them                  991  (16.33% = the test base rate)
recall recomputed FROM THE DATABASE      0.6135  (threshold.json says 0.6135)  OK
```

`build_features.py` also now rebuilds byte-identically (verified by sha256).

### Phase 3 — LLM Explanation Layer (built; needs a Groq key for prose)

- `explain.py` — one Groq call, stateless, single-turn. **Not a chatbot**: no history,
  no follow-ups, no tools, no routing by severity.
- Model `llama-3.3-70b-versatile`; cached in `risk_explanations` keyed on `score_id`.
- Cache stores `risk_probability_at_generation`, so a paragraph written against a score
  that has since moved is detected as **stale** rather than served.
- Backoff-and-retry on 429 (honours `Retry-After`, exponential + jitter) before any
  fallback. A fallback model is never substituted silently.
- Failure mode is a **missing paragraph, never a changed score** — verified.

**`test_explain.py` — 35 checks, no API key needed.** Every model call is stubbed,
which is the stronger test: the boundary must hold whatever a model returns.

Check 3 is the graded one. It feeds the layer a model whose every reply is an explicit
attempt to overturn the decision — JSON overrides, prose contradictions, injected
instructions, a literal `UPDATE risk_scores SET risk_probability=0.0` — and requires
the decision back bit-identical:

```
model demanded  risk_probability=0.0, recommendation=allow
system returned risk_probability=1.000000, recommendation=hold_payout
```

Hard rule #4 holds by construction, not by convention.

**Still needed:** a free `GROQ_API_KEY` in `.env` (console.groq.com/keys). Until then
`explain.py` runs its documented failure path — score intact, paragraph absent, reason
recorded — and `--invariance` already passes its identical-`risk_probability` half.

### Phase 4 — Streamlit Demo App (complete)

`app.py`, one file, three tabs. Run it with `streamlit run app.py`.

| Tab | What it shows |
|---|---|
| **Score an order** | Loads a real held-out order (highest risk / typical / a return the model caught / one it missed), then lets you edit any field. Score, band, recommendation, and a diverging bar chart of per-feature contributions. Explanation underneath. |
| **Cost & threshold** | **The demo moment.** Sliders for every cost input; the curve redraws and the optimal marker moves. |
| **Held-out evidence** | Metrics with the base rate beside each, confusion matrix, measured-vs-assumed cost inputs, and an explicit "what this does not measure" section. |

**The threshold tab is the argument.** Verified interactively: prevention 30%→90%
moves the optimum 0.18→0.09; review cost ×12 pushes it to 0.98 (flag almost nothing).
That is the case that 0.5 is a default, not a threshold — the operating point falls
out of what a mistake costs.

Moving a slider never silently re-deploys anything. When the slider-implied optimum
diverges from the shipped 0.18 the app says so, and says that changing the operating
point is a decision someone signs off on.

Honesty carried into the UI rather than left in the write-up: accuracy is shown with a
warning that it sits *below* the flag-nothing floor; cost inputs are tabulated as
**measured** or **assumed**; the cold-start `-1` sentinel is explained when it fires;
recommendations are labelled "a recommendation, not an action".

Charts follow a validated palette (CVD ΔE 21.6, normal-vision ΔE 32.3 on the diverging
pair), pick dark steps for dark mode rather than flipping, and never carry meaning by
colour alone.

Verified headlessly with `streamlit.testing.v1.AppTest`: 0 exceptions across all three
tabs, every example loader, the cold-start path, the cost sliders, and the explanation
button's no-key path.

### Audit — what a full review found and fixed

A pass over every file for vulnerabilities and correctness. Ordered by how much
each one distorted the numbers.

**1. The label horizon did not match the maturity horizon.** *(the big one)*
Maturity guaranteed every order 90 days of observation, but the label counted
returns at **any** horizon. So an order from Dec 2009 was watched for 667 days and
one from Sep 2011 for 90, and the positive rate rose with the length of the watch —
17.6% in the shortest-window quintile against 20.3% in the longest. That is exposure
time, not risk. Because the split is chronological, train sat in the long-window end
and test in the short one, **manufacturing a 1.21pp train/test gap out of nothing.**
Capping the label at the window maturity already guarantees cut that gap to 0.50pp
and moved the positive rate from 18.58% to 16.73%. `test_phase1.py` check 4 now fails
if the positive rate drifts across observation-window quintiles, with the tolerance
set at 0.02 — deliberately tight enough to have caught the old label's 0.027.

**2. Lookahead in the strongest feature.** `basket_sku_return_rate` was built from
train-period purchases and their *eventual* outcomes; **7.3% of those returns were
observed after the split date**, so a test order was scored with knowledge of returns
that had not happened when it was placed. The comment claimed "nothing from the test
window can inform it" — true of the purchases, false of their outcomes. Both sides are
now cut at the split boundary.

**3. `price_vs_sku_mean` averaged over the whole dataset**, pricing 474 SKUs entirely
from rows that did not exist yet and shifting 631 more by over 5%. Now built from
known-at-split-date purchases, with a global fallback for unseen SKUs.

> Removing 2 and 3 cost almost nothing: ROC-AUC 0.750 → 0.748. The leakage was small,
> which is the useful finding — the model was never relying on it.

**4. The same constant defined in six files.** `MIN_GAP_DAYS` and `RETURN_WINDOW_DAYS`
lived independently in `build_labels`, `build_features`, `build_database`,
`make_synthetic_dataset` and `test_phase1` — one of them under the comment *"must match
build_labels.py"*, which is a comment doing a constant's job. Now `config.py`, with a
single `genuine_returns()` applying the label window everywhere.

**5. `score_to_db.py` was not idempotent** despite saying so. Its delete list missed
`model_feature_importance` and `evaluations`, both of which reference `models(id)`, so
a second run against the same database died on a foreign key. It looked fine only
because every previous run happened to follow a fresh `build_database.py`.
`test_phase1.py` now derives the FK dependents from `sqlite_master` and fails if any
is unhandled. Related: `reviews` is a **human audit trail**, so the script now refuses
to run rather than deleting reviewer decisions to make room for a re-score.

**6. Band boundaries disagreed.** `score_to_db.py` wrote `min(threshold*2, 1.0)` into
`threshold_config` while `risk_band()` used an unclipped `threshold*2`. Identical at
the shipped 0.17, divergent above 0.5 — and the database would have been the one
telling the truth to anyone reading it with SQL. Both now call `predict.band_bounds()`.

**7. The source dataset was fetched with no integrity check.** A plain HTTPS pull of a
third-party `.rda` from a force-pushable branch, straight into `pyreadr`. Now pinned by
SHA-256 in `config.py` and verified on every load — downloaded to a temp name and
checked *before* it becomes the file the pipeline reads, so a corrupted fetch cannot be
cached as a good one.

**8. Secrets could reach the database and the UI.** `explain.py` writes error strings
into `risk_explanations.error_message` and renders them in Streamlit; an upstream client
can echo an API key back inside an authentication error. All error text now passes
through `redact()`, which strips bearer tokens, `gsk_`/`sk-` keys, and
`authorization`/`api_key` headers, and the live key value itself.

**9. A raw-HTML sink in the app.** The recommendation card used
`unsafe_allow_html=True`. Everything it interpolated came from closed dicts, so it was
safe as written — but it left an HTML sink one careless edit away from rendering a
model-authored or order-derived string. Replaced with native Streamlit callouts: same
colour semantics, no sink, and theme-aware for free.

**10. Smaller items.** SQL identifiers in the three unavoidable f-string spots are now
validated against `^[A-Za-z_][A-Za-z0-9_]*$` and quoted (defence in depth — the names
come from our own schema). A connection leak in the app's explanation path is closed in
a `finally`. A score-id collision check. A stale hardcoded base rate inside the LLM
prompt — in the one place designed to stop the model inventing numbers — now read from
the artefacts. `evaluate_model.py` no longer reloads the model from disk on every call.
The pickle/joblib trust boundary is documented in `predict.py`.

**Not a finding, stated for completeness:** `joblib.load` and `pd.read_pickle` execute
arbitrary code on untrusted input. The artefacts are locally generated and gitignored,
and the `.rda` they descend from is now checksummed, so the *input* to the chain is
verified. Do not add a "download the model" path without signing the artefact.

### Verification — all suites green

| Suite | Checks | Covers |
|---|---|---|
| `test_phase1.py` | 72 | pipeline, leakage, split, database, and now the score↔label join |
| `test_model.py` | 26 | artefact wiring, determinism, sentinels, threshold contract |
| `evaluate_model.py` | 27 (Part A) | edge cases; Part B is measurement, not pass/fail |
| `test_explain.py` | 35 | AI/non-AI boundary, adversarial model, cache, retry, failure mode |
| | **160** | |

Plus `app.py` exercised headlessly via `AppTest` — 0 exceptions across all tabs and
interaction paths.

### Other Files in Place

| File | Purpose |
|---|---|
| `README.md` | The public explainer — what it does, the cost argument, how to run it, and what it does not measure. Every number in it is verified against `artefacts/threshold.json`. **Carries no hackathon or sponsor framing**: the honest-metrics standard reads as self-imposed rather than externally graded, which is the stronger claim for a public repo. |
| `docs/*.png` | Committed copies of the two figures the README embeds. `artefacts/` is gitignored, so a README pointing at it would show broken images on GitHub. Refresh after a retrain: `cp artefacts/cost_vs_threshold.png artefacts/threshold_sensitivity.png docs/` |
| `AI_Risk_Manager_schema_v3.sql` | 30-table DDL (SQLite + Postgres compatible) |
| `AI_Risk_Manager_Approach.pdf` | Approach document |
| `AI_Risk_Manager_DB_Schema_FINAL.pdf` | Schema documentation |
| `CLAUDE.md` | Project context and hard rules. *Still carries the original hackathon framing — internal, but tracked, so it is visible on GitHub.* |
| `BUILD_PLAN_1.md` | Full build plan with all phases. *Same — still references the track.* |
| `NOTEBOOK_PROMPT.md` | The prompt used to generate the training notebook |
| `.env.example` | Environment template (Groq keys, paths) |
| `cost_model.py` | The one cost model |
| `predict.py` | The one scoring path |
| `score_to_db.py` | Joins Phase 2 into `risk.db` |
| `explain.py` | Phase 3 explanation layer |
| `requirements.txt` | All dependencies across phases 1–4 |
| `data/synthetic_features.csv` | Tracked CSV for Kaggle upload |

---

## What's NOT Done Yet ⬜

### Phase 5 — Write-up (4 of 5 delivered; one item blocked on a key)

**`README.md` delivers most of this.** It is the public explainer: what the model does,
why the threshold is 0.17 rather than 0.5, how to run everything, and — at length —
what the project does *not* measure.

- [x] Trained model with documented feature set
- [x] Held-out evaluation report — metrics with the base rate beside every number, and
      accuracy flagged as *below* the 0.837 do-nothing floor rather than buried
- [x] FP/FN cost model explanation, threshold-vs-cost chart *(both figures embedded
      from `docs/`)*
- [x] **AI / non-AI boundary write-up**, with the adversarial-model transcript as
      evidence
- [ ] **LLM explanation samples** — blocked on a `GROQ_API_KEY`. The harness is built
      and tested against stubs; `python explain.py --invariance` already passes its
      identical-`risk_probability` half, but no real paragraph exists yet, so the
      three-model invariance *table* cannot be filled in.

Every number in `README.md` is checked against `artefacts/threshold.json` and the
pickles rather than transcribed by hand — links resolve, metrics match to three
decimals, confusion-matrix cells match, and the three suite totals match what the
suites actually print.

### Optional — FastAPI

- [ ] Only if phases 3–5 are done
- [ ] 3 endpoints: `POST /risk/score`, `GET /risk/scores/{id}/explanation`, `GET /health`

---

## Kaggle Upload

The notebook to upload is **`notebooks/train_model.ipynb`**.

**Data dependency:** Upload `data/synthetic_features.csv` as a Kaggle dataset, then point the notebook's load path to the Kaggle input directory. The notebook has a flag to switch between real `features.pkl` and the synthetic fallback.

---

## Quick Start — Rebuilding Everything from Scratch

```powershell
cd c:\CODES\AI_risk_management_system
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # fill in GROQ_API_KEY

# Phase 1 — data pipeline
python build_labels.py        # ~1 min (downloads 5.5 MB)
python build_features.py      # ~30 s
python build_database.py      # ~20 s

python test_phase1.py         # 72 checks, must be all green

# Phase 2 — model
jupyter nbconvert --to notebook --execute --inplace notebooks/train_model.ipynb

# Verify
python test_model.py          # 26 invariant checks
python evaluate_model.py      # 27 edge cases + the held-out report

# Phase 2 -> database, then Phase 3
python score_to_db.py         # writes 6,070 scores; verifies recall in SQL
python test_explain.py        # 35 checks, no API key needed
python explain.py --sample 3  # needs GROQ_API_KEY in .env
python explain.py --invariance
```

---

## Key Decisions Already Made (Don't Re-litigate)

| Decision | Why |
|---|---|
| Return-risk, not fraud | Fraud is a different track |
| Returns only, chargebacks unevaluated | No public dataset carries disputes; synthetic labels would fail the honesty bar |
| UCI Online Retail II dataset | 1M+ rows, real returns via credit notes, real customer IDs |
| Amounts in GBP, not INR | UK retailer; converting would imply the data is Indian |
| Winner = Logistic Regression | LightGBM's edge was 0.2%, below the pre-committed 2% bar |
| Threshold = 0.17 | Cost-optimal, not F1-optimal and not 0.5 |
| One cost model, in `cost_model.py` | Two copies meant two different "optimal" thresholds |
| No SMOTE / resampling | 18% positive rate needs no rebalancing; rebalancing distorts calibrated probabilities |
| Chronological split only | Never `train_test_split`, never plain k-fold |

---

## Known Limitations (State These, Don't Hide)

- UK **wholesale gift retailer** — median order £304, customers mostly businesses (B2B return behaviour)
- 22.8% of source rows have no customer ID → excluded entirely
- No payment method, addresses, or discount data in the source
- Chargebacks are architecturally supported in the schema but **unevaluated**
