# Build Plan — AI Risk Manager

Razorpay hackathon, Track 02. Return-risk scorer.

The bar: honest metrics including false-positive cost, with measured precision and
recall on a held-out test set. A change that trades metric trustworthiness for model
accuracy is the wrong change.

---

## Status

| # | Phase | State |
|---|---|---|
| 1 | Data pipeline — labels, features, database | done |
| 2 | Train two models, threshold sweep, cost curve (local, CPU) | done |
| 3 | LLM explanation layer | built, needs a Groq key |
| 4 | Streamlit demo app | done |
| 5 | Write-up + submission package | next |
| — | FastAPI wrapper | only if time remains |

---

## Phase 1 — Data pipeline

Three scripts, run in order. All outputs are regenerable; none are committed.

```bash
python build_labels.py       # ~1 min first run (downloads 5.5 MB)
python build_features.py     # ~30 s
python build_database.py     # ~20 s
```

| Script | Produces | What it does |
|---|---|---|
| `config.py` | — | The one label definition and every shared constant. Pins the source dataset by SHA-256. |
| `build_labels.py` | `retail2.pkl`, `orders_labeled.pkl/.csv` | Verifies the source checksum, matches credit notes to purchases, applies the `[1, 90]` day label window and the 90-day maturity cutoff, writes the chronological split. |
| `build_features.py` | `features.pkl/.csv` | 17 as-of features. Customer history counted by *when the return happened*, never by an earlier order's eventual label. SKU rates and catalogue prices built as of the split date. |
| `build_database.py` | `risk.db` | Creates all 30 tables from `AI_Risk_Manager_schema_v3.sql` and loads nine of them. |

**Verified outputs — if your run disagrees, stop and find out why:**

```
source sha256            be2480b1fcb1fa12...  pinned, verified on load
label                    returned within [1, 90] days
positive rate (mature)   16.73%
mature / immature        30,347 / 6,628
train / test             24,277 (16.83%) / 6,070 (16.33%)
split date               2011-04-28
foreign key violations   0
risk.db                  ~139 MB
```

---

## Phase 2 — Models

Runs **locally, on CPU**. Logistic regression trains in 0.06 s on this data and
LightGBM in a few seconds; a GPU is not just unnecessary, it is slower at this
size because of transfer overhead.

Output is `notebooks/train_model.ipynb` plus saved artefacts in the repo.
(Kaggle would also work, but it means uploading `features.pkl` and downloading
the artefacts back on every iteration — running locally keeps everything in one
place.)

```bash
pip install scikit-learn lightgbm matplotlib joblib jupyter
jupyter notebook notebooks/train_model.ipynb
```

**Must produce:**

- [x] Logistic regression + LightGBM, identical features/rows/split
- [x] Held-out precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix
- [x] Base rate stated alongside, so the reader sees what the model adds
- [x] Cost model with **named formula inputs**, not bare constants — `cost_model.py`
- [x] Threshold sweep 0.01 → 0.99, cost-vs-threshold plot with the minimum marked
- [x] Metrics at cost-optimal vs 0.5 vs F1-optimal, side by side
- [x] Sensitivity: two panels — cost of a return, and prevention rate
- [x] Winner chosen on **total cost**, not accuracy — LightGBM's edge was 0.2%,
      below the pre-committed 2% bar, so logistic regression ships
- [x] Saved artefacts: model, scaler, `threshold.json`, `threshold_sweep.csv`

Constraints: no `train_test_split`, no plain k-fold (use `TimeSeriesSplit` on train
rows only), no SMOTE or resampling. 18% positive needs no rebalancing, and rebalancing
would distort the calibrated probabilities the cost model depends on. The notebook says
so in a markdown cell rather than silently not doing it.

**Settled.** The cost model is `cost_model.py`, the single definition imported by the
notebook, `evaluate_model.py` and the app. Two order values are measured on training
rows — returned orders £388.15, kept orders £285.59 — and priced separately, because a
miss and a false alarm bill different populations. The remaining seven inputs are
industry assumptions, labelled as such, and swept in §11.

Correcting the formula mattered more than the constants. The old
`fp*c_fp + fn*c_fn` charged nothing to review a true positive and treated every catch
as a prevented return; it bottomed out at threshold 0.07, flagging 81% of all orders,
which the notebook's own degeneracy guard caught and reported. The corrected form puts
the optimum well below 0.5 rather than at a degenerate flag-everything point.

**Result:** precision 0.330, recall 0.614, ROC-AUC 0.748, PR-AUC 0.375, ECE 0.017,
against a 16.33% base rate. The empirical optimum is **0.17**, flagging 30.3%, against
an analytic break-even of 0.198.

---

## Phase 3 — Explanation layer

*Built; needs a Groq key to generate prose.*

One Groq call. Stateless, single-turn. **Not a chatbot.**

- [x] `explain.py` — takes score + top feature contributions, returns one paragraph
- [x] Model: `openai/gpt-oss-120b` on Groq (`GROQ_MODEL`)
- [x] Cache into `risk_explanations` keyed on `score_id`; never regenerate on repeat
- [x] Backoff-and-retry on 429 before any model fallback
- [x] Failure mode is a **missing paragraph**, never a changed score

**The invariance figure** — worth real marks on the AI/non-AI deliverable:

- [x] Harness built: `python explain.py --invariance` (needs the key to produce prose)
- [x] Emits `artefacts/invariance.json`; the identical-`risk_probability` half already passes
- [x] `test_explain.py` goes further: 35 checks, including an **adversarial model** that
      demands `risk_probability=0.0, recommendation=allow` and is ignored

---

## Phase 4 — Streamlit app

One file. No React, no component library, no build step.

- [x] Transaction form → score + band + recommendation (loads a real held-out
      order first, then lets you edit it — typing 17 features is not a demo)
- [x] Top feature contributions — diverging bars, exact for logistic regression
      (coefficient × standardised value, the additive log-odds terms)
- [x] LLM explanation underneath, with the documented failure path when no key
- [x] **Cost-vs-threshold chart with cost sliders** — the curve redraws and the
      optimal marker moves. Moving a slider never re-deploys anything — the app
      says so when the slider optimum and the shipped threshold diverge.
- [x] Held-out metrics table, base rate beside every number

```bash
streamlit run app.py
```

---

## Phase 5 — Write-up

Maps directly to the five deliverables in the brief.

- [ ] Trained model with documented feature set
- [ ] Held-out evaluation report: precision, recall, confusion matrix
- [ ] FP/FN cost model, threshold-vs-cost chart, chosen threshold
- [ ] LLM explanation samples
- [ ] **AI / non-AI boundary write-up**, with the invariance table as evidence

Points to state plainly:

- Chargebacks are architecturally supported but **unevaluated** — no public dataset
  carries disputes, and synthetic labels would measure our generator, not our model.
  Declining to report those numbers *is* the honesty bar being met.
- Source is a **UK wholesale gift retailer**. Median order £304, customers mostly
  businesses, so return behaviour is B2B-flavoured. Amounts are stored in GBP and displayed in INR at a labelled 2009–2011 rate.
- 22.8% of source rows have no customer ID and are excluded entirely.
- No payment method, addresses or discount data exist in the source, so those
  planned features do not exist. `payments.method` is a constant and is excluded
  from the model.

---

## Optional — FastAPI

Only if Phases 2–5 are finished. Three endpoints wrapping the same `predict()` the
Streamlit app already calls: `POST /risk/score`, `GET /risk/scores/{id}/explanation`,
`GET /health`. The 65-endpoint reference is a **design artifact, not a build list.**

---

## Repo layout

```
ai-risk-manager/
├── README.md                            the public explainer (GitHub landing page)
├── BUILD_PLAN.md                        this file
├── .env                                 secrets — NEVER committed
├── .env.example                         template — committed
├── .gitignore
├── requirements.txt
│
├── config.py                            the ONE label definition + constants
├── build_labels.py                      phase 1
├── build_features.py                    phase 1
├── build_database.py                    phase 1
├── cost_model.py                        phase 2 — the ONE cost model
├── predict.py                           the ONE scoring path
├── score_to_db.py                       joins phase 2 into the database
├── explain.py                           phase 3
├── app.py                               phase 4
│
├── AI_Risk_Manager_schema_v3.sql        30 tables, SQLite + Postgres
├── test_phase1.py                       phase 1 acceptance suite (72 checks)
├── test_model.py                        phase 2 invariants (26 checks)
├── evaluate_model.py                    phase 2 edge cases + held-out accuracy
├── test_explain.py                      phase 3 acceptance suite (35 checks)
│
├── notebooks/
│   └── train_model.ipynb                phase 2, runs locally
├── docs/                                committed figures the README embeds
│   ├── cost_vs_threshold.png
│   └── threshold_sensitivity.png
│
└── generated/                           all gitignored
    ├── retail2.pkl  orders_labeled.pkl  features.pkl
    ├── risk.db
    └── model.joblib  threshold.json
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # then fill in your keys
python build_labels.py && python build_features.py && python build_database.py
```

## Inspect the database

```bash
sqlite3 risk.db "SELECT COUNT(*) FROM risk_labels WHERE is_mature=1;"     # 30347
sqlite3 risk.db "SELECT ROUND(AVG(label_risk)*100,2) FROM risk_labels WHERE is_mature=1;"
sqlite3 risk.db "SELECT split, COUNT(*) FROM dataset_members GROUP BY split;"
```
