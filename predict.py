"""
AI Risk Manager
The scoring path. One module, imported by everything that needs a score:
score_to_db.py, explain.py, the Streamlit app, and (if it gets built) FastAPI.

    from predict import Scorer
    s = Scorer()
    s.score_order({...17 features...})
    -> {"risk_probability": 0.42, "risk_band": "medium",
        "recommendation": "manual_review", ...}

WHY THIS EXISTS
---------------
The scoring logic used to live inline in test_model.py, with a second copy in
evaluate_model.py. One of them carried a comment saying "the single scoring
path; everything else in the repo should call this" - while sitting inside a
test file that nothing imports. Same failure as the cost model: the intent was
right and the wiring was not.

THE VOCABULARY IS THE SCHEMA'S, NOT OURS
----------------------------------------
`AI_Risk_Manager_schema_v3.sql` constrains what may be written:

    risk_band       IN ('low','medium','high')
    recommendation  IN ('allow','manual_review','hold_payout','request_verification')

The earlier inline version emitted 'approve' / 'review' / 'hold_for_review',
none of which are legal values. Nothing caught it because nothing had ever
written a score to the database. Every recommendation this module returns is a
value the CHECK constraint accepts.

HARD RULE #4 - THE LLM NEVER DECIDES
------------------------------------
This module is the only thing that produces `risk_probability`, `risk_band` and
`recommendation`. `explain.py` reads a finished score and writes prose. There is
no code path from generated text back into any field here, and there must not
be one - it is a graded deliverable.

HARD RULE #6 - HUMAN IN THE LOOP
--------------------------------
What comes back is a RECOMMENDATION. Acting on it needs a named reviewer_id in
the `reviews` table. Nothing here takes an action.

TRUST BOUNDARY
--------------
`joblib.load` unpickles, and unpickling executes arbitrary code. The artefacts
in artefacts/ are produced locally by notebooks/train_model.ipynb and are
gitignored, so the file this loads is one this machine wrote - but if you ever
accept a model.joblib from somewhere else, that is code execution, not data
loading. Do not add a "download the model" path without signing the artefact.

The same applies to `pd.read_pickle` on features.pkl and retail2.pkl. The .rda
that everything descends from IS pinned by SHA-256 in config.py, so the input to
the chain is verified even though the intermediates are not.
"""

import json
import os

import joblib
import numpy as np

# config is imported for its side effect as well as its constants: it loads
# .env, so MODEL_PATH / THRESHOLD_PATH / SCALER_PATH honour it.
from config import ART_DIR, MODEL_PATH, SCALER_PATH, THRESHOLD_PATH

ROOT = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------- bands
# Bands are defined RELATIVE to the operating threshold, so they move with it.
# A band is not an independent opinion about risk; it is a restatement of where
# the score sits against the point the cost model chose.
BAND_LOW, BAND_MEDIUM, BAND_HIGH = "low", "medium", "high"
HIGH_BAND_MULTIPLIER = 2.0

# Schema vocabulary. See the CHECK constraint on risk_scores.recommendation.
RECOMMENDATION = {
    BAND_LOW:    "allow",
    BAND_MEDIUM: "manual_review",
    BAND_HIGH:   "hold_payout",
}


def band_bounds(threshold):
    """
    (medium_starts_at, high_starts_at).

    One expression, because score_to_db.py wrote `min(threshold*2, 1.0)` into
    threshold_config while risk_band() used an unclipped `threshold*2`. At the
    shipped 0.17 they agree, but above 0.5 the database and the scorer would
    have disagreed about where the high band begins - and the database would
    have been the one telling the truth to anyone reading it with SQL.
    """
    return float(threshold), float(min(threshold * HIGH_BAND_MULTIPLIER, 1.0))


def risk_band(p, threshold):
    """Bands sit at the threshold and at twice the threshold."""
    medium_at, high_at = band_bounds(threshold)
    if p >= high_at:
        return BAND_HIGH
    if p >= medium_at:
        return BAND_MEDIUM
    return BAND_LOW


class Scorer:
    """Loads the artefacts once and scores against them."""

    def __init__(self, art_dir=None):
        # art_dir stays supported (tests point it at a temp directory); when it
        # is not given the env-overridable paths from config are used.
        self.art_dir = art_dir or ART_DIR
        meta_path = (os.path.join(art_dir, "threshold.json") if art_dir
                     else THRESHOLD_PATH)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"{meta_path} not found. Run notebooks/train_model.ipynb first.")
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.features = self.meta["features"]
        self.threshold = float(self.meta["chosen_threshold"])
        self.requires_scaler = bool(self.meta["requires_scaler"])
        self.model = joblib.load(os.path.join(art_dir, "model.joblib")
                                 if art_dir else MODEL_PATH)

        scaler_path = (os.path.join(art_dir, "scaler.joblib") if art_dir
                       else SCALER_PATH)
        self.scaler = (joblib.load(scaler_path)
                       if self.requires_scaler and os.path.exists(scaler_path) else None)
        if self.requires_scaler and self.scaler is None:
            raise FileNotFoundError(
                "threshold.json says requires_scaler=true but scaler.joblib is missing.")

        # Artefacts trained on synthetic data must never be mistaken for real
        # ones. Callers that report metrics should check this.
        self.reportable = bool(self.meta.get("REPORTABLE", False))

    # ------------------------------------------------------------- internals
    def _matrix(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != len(self.features):
            raise ValueError(
                f"expected {len(self.features)} features, got {X.shape[1]}")
        if not np.isfinite(X).all():
            # Rejecting is deliberate. A NaN silently scoring as 0.5 is worse
            # than an exception the caller has to handle.
            raise ValueError("feature matrix contains NaN or infinity")
        return X

    def _to_row(self, order: dict):
        missing = [f for f in self.features if f not in order]
        if missing:
            raise KeyError(f"missing features: {missing}")
        # Built by NAME, so caller key order is irrelevant. Positional callers
        # must supply the list in self.features order.
        return np.array([[float(order[f]) for f in self.features]])

    # ---------------------------------------------------------------- public
    def score_batch(self, X):
        """Probabilities for a 2-D array of feature rows, in `features` order."""
        X = self._matrix(X)
        Xs = self.scaler.transform(X) if self.requires_scaler else X
        return self.model.predict_proba(Xs)[:, 1]

    def contributions(self, order: dict, top_n=None):
        """
        Signed per-feature contribution to this score.

        For logistic regression this is exact: coefficient x standardised
        value, the additive terms of the log-odds. That is the whole reason a
        linear model was worth keeping - the explanation is the arithmetic, not
        a post-hoc approximation of it.

        For a tree model there is no such decomposition, so this returns None
        rather than inventing one.
        """
        if not hasattr(self.model, "coef_"):
            return None
        X = self._matrix(self._to_row(order))
        Xs = self.scaler.transform(X) if self.requires_scaler else X
        terms = self.model.coef_[0] * Xs[0]
        out = [{"feature": f, "value": float(X[0][i]),
                "contribution": float(terms[i])}
               for i, f in enumerate(self.features)]
        out.sort(key=lambda d: abs(d["contribution"]), reverse=True)
        return out[:top_n] if top_n else out

    def score_order(self, order: dict, top_n=5):
        """
        Score one order given as a plain dict of the 17 features.

        Returns a FINISHED decision. No downstream layer - the LLM included -
        may alter any field in it.
        """
        p = float(self.score_batch(self._to_row(order))[0])
        band = risk_band(p, self.threshold)
        return {
            "risk_probability": p,
            "risk_band": band,
            "recommendation": RECOMMENDATION[band],
            "flagged": bool(p >= self.threshold),
            "threshold_applied": self.threshold,
            # The cold-start path is part of the decision's provenance: a score
            # built on no customer history should be read differently from one
            # built on fifty prior orders.
            "customer_history": ("none" if float(order.get("is_new_customer", 0)) == 1
                                 else "present"),
            "top_features": self.contributions(order, top_n=top_n),
        }


_DEFAULT = None


def get_scorer():
    """Process-wide singleton, so Streamlit does not reload joblib per rerun."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Scorer()
    return _DEFAULT


if __name__ == "__main__":
    import pandas as pd

    s = Scorer()
    print(f"model      {s.meta['winner']}")
    lo, hi = band_bounds(s.threshold)
    print(f"threshold  {s.threshold}   (bands at {lo} and {hi})")
    print(f"features   {len(s.features)}")
    print(f"reportable {s.reportable}")

    df = pd.read_pickle(os.path.join(ROOT, "features.pkl"))
    test = df[df.split == "test"]
    row = test.iloc[len(test) // 2]
    out = s.score_order({f: row[f] for f in s.features})

    print(f"\nworked example - order {row.Invoice} "
          f"(actual outcome: {'returned' if row.returned else 'kept'})")
    for k in ("risk_probability", "risk_band", "recommendation",
              "flagged", "customer_history"):
        print(f"  {k:<20} {out[k]}")
    print("  top contributions")
    for c in out["top_features"]:
        print(f"    {c['feature']:<28} value {c['value']:>12,.3f}"
              f"   contribution {c['contribution']:>+7.3f}")
