"""
Step 4: score the held-out orders and write them into risk.db. Run after
build_database.py and the training notebook.

    python score_to_db.py

The database and the model were built separately and nothing joined them:
`models`, `threshold_config`, `risk_scores` and `risk_score_features` were all
empty, so the scoring half of a 30-table schema was decoration. That gap matters
downstream too - risk_explanations.score_id is a foreign key onto
risk_scores(id), so the explanation cache has nothing to key on until scores
exist.

Writes:

    models                1    the shipped model, with its feature list
    threshold_config      1    the operating point, marked is_current
    risk_scores       6,070    one per held-out order
    risk_score_features         17 per score - the exact vector the model saw,
                                with each feature's signed contribution

Test split only. Writing train scores would put numbers the model has already
seen into the database, and sooner or later someone computes a metric over the
whole table and reports it.

Idempotent: it clears the four tables it owns and rewrites them, so re-running
after a retrain does not accumulate stale scores.
"""

import hashlib
import json
import os
import re
import sqlite3
import time

import numpy as np
import pandas as pd

from predict import RECOMMENDATION, Scorer, band_bounds, risk_band

from config import DATASET_ID, DB_PATH, FEATURES_PKL, MERCHANT_ID

ROOT = os.path.dirname(os.path.abspath(__file__))
DB = DB_PATH

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def q(ident):
    """Validate-and-quote an identifier that cannot be a bound parameter."""
    if not _IDENT.match(ident):
        raise ValueError(f"refusing to interpolate {ident!r} into SQL")
    return f'"{ident}"'

# Deterministic ids: the same artefacts and the same data produce the same
# score ids on every run. Random ids would make the explanation cache useless,
# because a rebuild would orphan every cached paragraph.
def _id(prefix, *parts):
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return f"{prefix}_{h[:12]}"


ALGORITHM = {"logistic regression": "logistic_regression"}


def main():
    if not os.path.exists(DB):
        raise SystemExit("risk.db not found. Run build_database.py first.")

    scorer = Scorer()
    meta = scorer.meta

    if not scorer.reportable:
        print("  WARNING: artefacts are flagged NOT reportable (synthetic data).")
        print("  Scores will be written, but no metric over them is meaningful.")

    algorithm = ALGORITHM.get(meta["winner"])
    if algorithm is None:
        raise SystemExit(
            f"winner {meta['winner']!r} has no schema `algorithm` value. "
            f"The CHECK constraint allows logistic_regression/lightgbm/xgboost.")

    df = pd.read_pickle(FEATURES_PKL)
    test = df[df.split == "test"].sort_values("order_date").reset_index(drop=True)

    con = sqlite3.connect(DB)
    con.execute("PRAGMA foreign_keys=ON")
    now = int(time.time())

    # `reviews` is the human-in-the-loop audit trail: a named reviewer's
    # decision on a score. Rebuilding scores would orphan those rows, and
    # deleting them to make room is destroying an audit record to save a
    # re-run. Refuse instead, and make the operator say what should happen.
    n_reviews = con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    if n_reviews:
        con.close()
        raise SystemExit(
            f"{n_reviews:,} row(s) in `reviews` reference the scores this script "
            f"would replace. Those are human decisions, not derived data, so "
            f"they are not deleted automatically. Export them first, or run "
            f"build_database.py for a clean rebuild if they are disposable.")

    # Idempotent: drop what this script owns, children before parents.
    #
    # This list was wrong: `model_feature_importance` and `evaluations` both
    # reference models(id) and were missing, so a second run against the same
    # database died on a foreign key constraint. It looked idempotent only
    # because every earlier run happened to follow a fresh build_database.py.
    # Every dependent of the four tables below is now accounted for:
    #
    #   risk_scores       <- risk_score_features, risk_explanations, reviews
    #   models            <- model_feature_importance, evaluations, risk_scores
    #   threshold_config  <- risk_scores
    #
    # risk_explanations goes too: those paragraphs were written against scores
    # that will no longer exist.
    for t in ("risk_explanations", "risk_score_features", "risk_scores",
              "model_feature_importance", "evaluations",
              "threshold_config", "models"):
        con.execute(f"DELETE FROM {q(t)}")

    # --- the model
    model_id = _id("mdl", algorithm, meta["chosen_threshold"],
                   ",".join(meta["features"]))
    train_until = int(pd.Timestamp(meta["holdout"]["split_date"]).timestamp())
    con.execute(
        "INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (model_id, MERCHANT_ID, algorithm, DATASET_ID,
         json.dumps({"selection_rule": meta["selection_rule"],
                     "rationale": meta["rationale"]}),
         json.dumps(meta["features"]), train_until, "active",
         "artefacts/model.joblib", now, now, now))

    # Standardised coefficients are the model's own importances - no surrogate.
    if hasattr(scorer.model, "coef_"):
        con.executemany(
            "INSERT INTO model_feature_importance VALUES (?,?,?,?)",
            [(model_id, f, float(c), r + 1)
             for r, (f, c) in enumerate(sorted(
                 zip(meta["features"], scorer.model.coef_[0]),
                 key=lambda t: abs(t[1]), reverse=True))])

    # --- the threshold
    # threshold_low is the operating point; threshold_high is where the high
    # band starts. Both come from predict.py so the database cannot disagree
    # with the scorer about where a band boundary is.
    thr_id = _id("thr", meta["chosen_threshold"], model_id)
    medium_at, high_at = band_bounds(scorer.threshold)
    con.execute(
        "INSERT INTO threshold_config VALUES (?,?,?,?,?,?,?,?,?,?)",
        (thr_id, MERCHANT_ID, medium_at, high_at,
         "optimized", None,
         f"Minimum total cost on the held-out test set. {meta['rationale']} "
         f"Analytic break-even {meta.get('analytic_break_even')}.",
         None, 1, now))

    # --- the scores
    X = test[meta["features"]].to_numpy(float)
    t0 = time.perf_counter()
    p = scorer.score_batch(X)
    latency_ms = int(round((time.perf_counter() - t0) * 1000 / max(len(test), 1)))

    Xs = scorer.scaler.transform(X) if scorer.requires_scaler else X
    coef = scorer.model.coef_[0] if hasattr(scorer.model, "coef_") else None

    score_rows, feat_rows = [], []
    for i, r in enumerate(test.itertuples()):
        prob = float(p[i])
        band = risk_band(prob, scorer.threshold)
        sid = _id("score", model_id, r.Invoice)
        score_rows.append((
            sid, MERCHANT_ID, f"pay_{r.Invoice}", f"order_{r.Invoice}",
            f"cust_{int(r.customer_id)}", model_id, prob, band,
            scorer.threshold, thr_id, RECOMMENDATION[band],
            "none" if int(r.is_new_customer) == 1 else "present",
            None, latency_ms, int(pd.Timestamp(r.order_date).timestamp())))
        if coef is not None:
            contrib = coef * Xs[i]
            feat_rows.extend(
                (sid, f, float(X[i][j]), float(contrib[j]))
                for j, f in enumerate(meta["features"]))

    # 48 bits of id is ample for 6k rows, but a silent collision would drop a
    # score and every downstream count would still look plausible.
    ids = [r[0] for r in score_rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("score id collision - widen the hash in _id()")

    con.executemany(
        "INSERT INTO risk_scores VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", score_rows)
    con.executemany(
        "INSERT INTO risk_score_features VALUES (?,?,?,?)", feat_rows)
    con.commit()

    # --- verify
    print(f"\n{'='*62}\n  SCORES WRITTEN TO risk.db\n{'='*62}")
    print(f"  model            {model_id}  ({algorithm})")
    print(f"  threshold        {thr_id}  @ {scorer.threshold}")
    for t in ("models", "threshold_config", "risk_scores",
              "risk_score_features", "model_feature_importance"):
        n = con.execute(f"SELECT COUNT(*) FROM {q(t)}").fetchone()[0]
        print(f"  {t:<26} {n:>9,}")

    bad = con.execute("PRAGMA foreign_key_check").fetchall()
    print(f"\n  foreign key violations   {len(bad)}")

    band_rows = con.execute(
        "SELECT risk_band, recommendation, COUNT(*), ROUND(AVG(risk_probability),4) "
        "FROM risk_scores GROUP BY risk_band, recommendation "
        "ORDER BY AVG(risk_probability)").fetchall()
    print("\n  band            recommendation        n      mean p")
    for b, rec, n, mp in band_rows:
        print(f"  {b:<15} {rec:<18} {n:>6,}    {mp:.4f}")

    # The database must agree with the label table it was built from. This is
    # the join that proves Phase 1 and Phase 2 are actually connected rather
    # than merely both present.
    agree = con.execute("""
        SELECT COUNT(*),
               SUM(CASE WHEN (s.risk_probability >= s.threshold_applied)
                         AND l.label_risk = 1 THEN 1 ELSE 0 END),
               SUM(l.label_risk)
        FROM risk_scores s
        JOIN risk_labels l ON l.payment_id = s.payment_id
        JOIN dataset_members m ON m.payment_id = s.payment_id AND m.split = 'test'
    """).fetchone()
    n, tp, pos = agree
    print(f"\n  scores joined to labels via payment_id   {n:,}")
    print(f"  actual returns among them                {pos:,}  "
          f"({pos/n:.2%} - must match the "
          f"{meta['holdout']['base_rate_test']:.2%} test base rate)")
    print(f"  flagged AND returned (true positives)    {tp:,}")

    recall = tp / pos
    expected = meta["holdout"]["recall"]
    ok = abs(recall - expected) < 0.005
    print(f"  recall recomputed FROM THE DATABASE      {recall:.4f}  "
          f"(threshold.json says {expected:.4f})  {'OK' if ok else 'MISMATCH'}")
    con.close()
    if not ok:
        raise SystemExit("database scores disagree with the saved metrics.")
    print(f"\n  Phase 1 and Phase 2 are joined: a metric computed by SQL over")
    print(f"  risk_scores x risk_labels reproduces the notebook's number.\n")


if __name__ == "__main__":
    main()
