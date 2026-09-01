"""
AI Risk Manager
Phase 2 model test harness. Runs the SAVED artefacts against the held-out split.

    python test_model.py

Run AFTER notebooks/train_model.ipynb has written artefacts/.

WHAT THIS TESTS
---------------
Invariants, not accuracy. Every check below is a property that must hold whatever
the data is, so the same suite covers the real features.pkl and the synthetic
fallback. The data source follows the artefacts: whatever threshold.json says it
was trained on is what gets scored here. Nothing re-trains anything.

  1  artefact contract   - required files exist and agree on the feature list
  2  score validity      - probabilities finite and inside [0, 1]
  3  determinism         - identical input scores identically, twice
  4  row independence    - a row's score does not depend on its neighbours
  5  sentinel handling   - -1 (no history) is not treated as 0 (never returned)
  6  threshold contract  - the flag decision is exactly p >= chosen_threshold
  7  band monotonicity   - higher probability never yields a lower risk band
  8  no label at inference - scoring touches none of the outcome columns
  9  metric reproduction - re-scoring reproduces the metrics in threshold.json
 10  honesty flags       - synthetic artefacts are marked non-reportable

WHAT THIS DOES NOT TEST
-----------------------
Whether the model is any good. If threshold.json reports DATA_IS_SYNTHETIC, the
precision and recall reproduced here measure the generator rather than the model,
and the summary says so. See make_synthetic_dataset.py.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(ROOT, "artefacts")
REAL_DATA = os.path.join(ROOT, "features.pkl")
SYNTHETIC_DATA = os.path.join(ROOT, "data", "synthetic_features.pkl")
ID_COL = "Invoice"

# Bands, recommendations and the scoring path all come from predict.py. They
# used to be redefined here, in a test file nothing imports, under a comment
# claiming to be "the single scoring path" - and the recommendations they
# emitted ('approve'/'review'/'hold_for_review') were not even legal values
# under the schema's CHECK constraint. A test that defines its own copy of the
# thing it is testing verifies nothing.
from predict import (  # noqa: E402
    BAND_LOW, BAND_MEDIUM, BAND_HIGH, RECOMMENDATION, Scorer, risk_band,
)

_PASS, _FAIL = [], []


def check(name, condition, detail=""):
    (_PASS if condition else _FAIL).append(name)
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"\n         {detail}" if detail and not condition else ""))
    return bool(condition)


# ---------------------------------------------------------------- scoring API
# Thin adapters so the existing checks keep their signatures. The work happens
# in predict.Scorer; nothing below reimplements it.
_SCORER = None


def _scorer():
    global _SCORER
    if _SCORER is None:
        _SCORER = Scorer(ART_DIR)
    return _SCORER


def score(model, scaler, X, needs_scaler):
    return _scorer().score_batch(X)


def score_one(model, scaler, meta, order: dict):
    return _scorer().score_order(order)


def main():
    print("=" * 72)
    print("  Phase 2 model test harness")
    print("=" * 72)

    # -------------------------------------------------- 1 artefact contract
    print("\n1 · artefact contract")
    required = ["model.joblib", "threshold.json"]
    for f in required:
        if not check(f"{f} exists", os.path.exists(os.path.join(ART_DIR, f))):
            print(f"\nCannot continue. Run notebooks/train_model.ipynb first "
                  f"(it writes to {ART_DIR}/).")
            return 1

    meta = json.load(open(os.path.join(ART_DIR, "threshold.json")))
    model = joblib.load(os.path.join(ART_DIR, "model.joblib"))
    scaler_path = os.path.join(ART_DIR, "scaler.joblib")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    features = meta["features"]
    thr = meta["chosen_threshold"]
    needs_scaler = meta["requires_scaler"]

    check("feature list has 17 entries", len(features) == 17, f"got {len(features)}")
    check("threshold inside (0, 1)", 0.0 < thr < 1.0, f"got {thr}")
    check("scaler present when required", (not needs_scaler) or scaler is not None)
    check("winner recorded", bool(meta.get("winner")))
    check("selection rule recorded", "cost" in meta.get("selection_rule", "").lower())

    print(f"\n         winner     {meta['winner']}")
    print(f"         threshold  {thr}")
    print(f"         device     {meta.get('training_device')}")

    # -------------------------------------------------- data
    # Score against whatever the artefacts were trained on, so the metric
    # reproduction check in section 9 is meaningful.
    is_syn = bool(meta.get("DATA_IS_SYNTHETIC"))
    DATA = SYNTHETIC_DATA if is_syn else REAL_DATA
    if not os.path.exists(DATA):
        hint = ("python make_synthetic_dataset.py" if is_syn
                else "python build_labels.py && python build_features.py")
        print(f"\n{DATA} not found. Run: {hint}")
        return 1
    df = (pd.read_pickle(DATA) if DATA.endswith(".pkl")
          else pd.read_csv(DATA, parse_dates=["order_date"]))
    print(f"         data source: {os.path.relpath(DATA, ROOT)}"
          f"  ({'synthetic' if is_syn else 'real'})")
    test = df[df.split == "test"].sort_values("order_date").reset_index(drop=True)
    X = test[features].to_numpy(float)
    y = test["returned"].to_numpy(int)
    print(f"         scoring {len(test):,} held-out orders\n")

    p = score(model, scaler, X, needs_scaler)

    # -------------------------------------------------- 2 score validity
    print("\n2 · score validity")
    check("no NaN or inf", bool(np.isfinite(p).all()))
    check("all probabilities in [0, 1]", bool((p >= 0).all() and (p <= 1).all()),
          f"min {p.min():.4f} max {p.max():.4f}")
    check("scores are not constant", float(p.std()) > 1e-6, f"std {p.std():.2e}")

    # -------------------------------------------------- 3 determinism
    print("\n3 · determinism")
    p2 = score(model, scaler, X, needs_scaler)
    check("same input scores identically twice", bool(np.array_equal(p, p2)))

    # -------------------------------------------------- 4 row independence
    print("\n4 · row independence")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(X))
    p_shuf = score(model, scaler, X[order], needs_scaler)
    check("shuffling rows does not change per-row scores",
          bool(np.allclose(p_shuf, p[order], atol=1e-12)))

    single = np.array([score(model, scaler, X[i:i + 1], needs_scaler)[0] for i in range(50)])
    check("scoring one row matches scoring the batch",
          bool(np.allclose(single, p[:50], atol=1e-10)))

    # -------------------------------------------------- 5 sentinel handling
    print("\n5 · sentinel handling (-1 means no history, not zero)")
    sent_col = "customer_prior_return_rate"
    i_sent = features.index(sent_col)
    cold = test[test[sent_col] == -1]
    check(f"{sent_col} = -1 rows exist in the test split", len(cold) > 0,
          "sentinel may have been imputed away upstream")

    probe = X[:400].copy()
    a = probe.copy(); a[:, i_sent] = -1.0
    b = probe.copy(); b[:, i_sent] = 0.0
    pa, pb = score(model, scaler, a, needs_scaler), score(model, scaler, b, needs_scaler)
    check("-1 and 0 produce different scores",
          not bool(np.allclose(pa, pb, atol=1e-9)),
          "the model treats 'no history' and 'never returned' identically")
    print(f"         mean |score(-1) - score(0)| = {np.abs(pa - pb).mean():.6f}")

    # -------------------------------------------------- 6 threshold contract
    print("\n6 · threshold contract")
    flagged = p >= thr
    check("flag decision is exactly p >= threshold",
          bool(np.array_equal(flagged, np.array([score_one(
              model, scaler, meta, dict(zip(features, row)))["flagged"]
              for row in X[:200]] + list(flagged[200:])))))
    check("flag rate is not 0% or 100%", 0.0 < flagged.mean() < 1.0,
          f"flag rate {flagged.mean():.4f} - degenerate operating point")

    # -------------------------------------------------- 7 band monotonicity
    print("\n7 · band monotonicity")
    rank = {BAND_LOW: 0, BAND_MEDIUM: 1, BAND_HIGH: 2}
    ordered = np.sort(p)
    bands = [rank[risk_band(v, thr)] for v in ordered]
    check("higher probability never yields a lower band",
          all(bands[i] <= bands[i + 1] for i in range(len(bands) - 1)))
    for band in (BAND_LOW, BAND_MEDIUM, BAND_HIGH):
        n = sum(1 for v in p if risk_band(v, thr) == band)
        print(f"         {band:<8} {n:>7,}  ({n/len(p)*100:5.1f}%)  -> {RECOMMENDATION[band]}")

    # -------------------------------------------------- 8 no label at inference
    print("\n8 · no outcome column reaches the model")
    leaky = {"returned", "return_date", "is_mature", "split", ID_COL, "customer_id"}
    check("no outcome/identifier column is in the feature list",
          not (leaky & set(features)), f"leaky: {sorted(leaky & set(features))}")

    stripped = test[features].copy()
    p_stripped = score(model, scaler, stripped.to_numpy(float), needs_scaler)
    check("scoring works with outcome columns absent entirely",
          bool(np.allclose(p_stripped, p, atol=1e-12)))

    # -------------------------------------------------- 9 metric reproduction
    print("\n9 · metric reproduction against threshold.json")
    rec = meta.get("holdout", {})
    auc = roc_auc_score(y, p)
    prec = precision_score(y, flagged, zero_division=0)
    ric = recall_score(y, flagged, zero_division=0)
    for label, live, saved in [("roc_auc", auc, rec.get("roc_auc")),
                               ("precision", prec, rec.get("precision")),
                               ("recall", ric, rec.get("recall"))]:
        if saved is None:
            check(f"{label} recorded in threshold.json", False)
            continue
        check(f"{label} reproduces ({live:.4f} vs {saved:.4f})",
              abs(live - saved) < 1e-3, f"drift {abs(live - saved):.6f}")

    check("recorded test row count matches", rec.get("n_test") == len(test),
          f"json {rec.get('n_test')} vs data {len(test)}")

    # -------------------------------------------------- 10 honesty flags
    print("\n10 · honesty flags")
    check("artefacts declare whether the data was synthetic",
          "DATA_IS_SYNTHETIC" in meta)
    check("artefacts carry a REPORTABLE flag", "REPORTABLE" in meta)
    if meta.get("DATA_IS_SYNTHETIC"):
        check("synthetic artefacts are marked NOT reportable",
              meta.get("REPORTABLE") is False)

    # -------------------------------------------------- worked example
    print("\n" + "-" * 72)
    print("  worked example - single-order scoring API")
    print("-" * 72)
    for idx in (int(np.argmin(p)), int(np.argmax(p))):
        row = dict(zip(features, X[idx]))
        out = score_one(model, scaler, meta, row)
        print(f"\n  order {test[ID_COL].iloc[idx]}  "
              f"(actual outcome: {'RETURNED' if y[idx] else 'kept'})")
        print(f"    risk_probability  {out['risk_probability']}")
        print(f"    risk_band         {out['risk_band']}")
        print(f"    recommendation    {out['recommendation']}")

    # -------------------------------------------------- summary
    print("\n" + "=" * 72)
    print(f"  {len(_PASS)} passed, {len(_FAIL)} failed")
    if _FAIL:
        print("\n  failed checks:")
        for f in _FAIL:
            print(f"    - {f}")
    if meta.get("DATA_IS_SYNTHETIC"):
        print("\n  *** These artefacts were trained on SYNTHETIC data. ***")
        print("  *** The checks above verify wiring and invariants only.  ***")
        print("  *** No metric here is reportable. See CLAUDE.md.         ***")
    if meta.get("degenerate_optimum"):
        print("\n  NOTE: threshold.json records a DEGENERATE cost optimum - the")
        print("  operating point is set by the FP:FN assumption, not the model.")
        print("  Revisit the inputs in cost_model.py before trusting the threshold.")
    print("=" * 72)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
