"""
Flask API behind the Vercel deployment. The Python runtime looks for a
top-level `app`, so that is what this module exports.

    GET  /                  the HTML page (/static/* is served by Flask)
    GET  /api/meta          model metadata from threshold.json
    POST /api/score         score one order (17 features in, decision out)
    GET  /api/sweep         pre-computed threshold sweep, shipped model only
    POST /api/cost-optimal  re-cost that sweep under supplied assumptions

Everything the model or the cost model decides is decided elsewhere: predict.py
owns risk_probability / risk_band / recommendation, cost_model.py owns every
currency conversion and every cost term. This module marshals JSON.

If the model fails to load the routes answer 503 with the reason rather than
letting an import-time exception become a bodyless 500.
"""

import csv
import os
import sys

from flask import Flask, jsonify, request, send_from_directory

# project root is one level up from api/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# The decision comes from predict.py and the money from cost_model.py. Nothing
# in this file computes either - it converts JSON to arguments and back.
from predict import Scorer, band_bounds  # noqa: E402
import cost_model as cm  # noqa: E402

app = Flask(__name__, static_folder=os.path.join(ROOT, "static"))

# Loaded once per cold start. Everything involved is tiny: logreg ~1KB,
# scaler ~1KB, threshold.json ~2.5KB.
#
# Catching Exception rather than FileNotFoundError on purpose: a truncated
# joblib or a threshold.json missing a key raises something else entirely, and
# an exception here is an exception at import time, which on a serverless
# runtime is a bare 500 with no JSON body to explain it. Every route already
# checks MODEL_LOADED and answers with the reason.
try:
    scorer = Scorer()
    MODEL_LOADED = True
    MODEL_ERROR = None
except Exception as e:  # noqa: BLE001
    scorer = None
    MODEL_LOADED = False
    MODEL_ERROR = f"{type(e).__name__}: {e}"


def _load_sweep(path, winner):
    """
    The sweep CSV holds one row per threshold PER CANDIDATE MODEL - 99 for
    logistic regression and 99 for LightGBM, in one file with a `model` column.

    Reading it unfiltered mixed both curves: the thresholds ran 0.01..0.99 and
    then started over, so the plotted line doubled back on itself, and the
    minimum over the mixed rows was LightGBM's optimum (0.19) reported as the
    operating point of a shipped logistic regression (0.17). That is a metric
    attributed to a model that is not deployed, which is the one thing this
    repo is not allowed to do. Keep only the shipped model's rows.
    """
    rows = []
    if not os.path.exists(path):
        return rows, f"{os.path.basename(path)} not found"
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            if row.get("model") != winner:
                continue
            rows.append({
                "threshold": float(row["threshold"]),
                "total_cost_pence": float(row["total_cost_pence"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
                "flag_rate": float(row["flag_rate"]),
                "tp": int(row["tp"]),
                "fp": int(row["fp"]),
                "fn": int(row["fn"]),
                "tn": int(row["tn"]),
            })
    rows.sort(key=lambda r: r["threshold"])
    if not rows:
        # A sweep with no row for the shipped model is a stale artefact. Serving
        # nothing is honest; serving another model's curve is not.
        return [], f"threshold_sweep.csv has no rows for the shipped model {winner!r}"
    return rows, None


# No shipped model means no model to attribute a curve to, so the sweep is not
# loaded at all rather than loaded unattributed.
if MODEL_LOADED:
    SWEEP_DATA, SWEEP_ERROR = _load_sweep(
        os.path.join(ROOT, "artefacts", "threshold_sweep.csv"),
        scorer.meta.get("winner"))
else:
    SWEEP_DATA, SWEEP_ERROR = [], MODEL_ERROR

# Total positives in the held-out set: tp+fn is constant across thresholds, so
# any row carries it. Flagging nothing means every positive is a miss.
TOTAL_POSITIVES = (SWEEP_DATA[0]["tp"] + SWEEP_DATA[0]["fn"]) if SWEEP_DATA else 0


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/meta", methods=["GET"])
def meta():
    """Return model metadata for the frontend."""
    if not MODEL_LOADED:
        return jsonify({"error": MODEL_ERROR}), 503

    meta = scorer.meta
    holdout = meta.get("holdout", {})
    bounds = band_bounds(scorer.threshold)

    return jsonify({
        "model_loaded": True,
        "winner": meta["winner"],
        "threshold": scorer.threshold,
        "features": scorer.features,
        "reportable": scorer.reportable,
        "band_bounds": {"medium": bounds[0], "high": bounds[1]},
        "holdout": holdout,
        "analytic_break_even": meta.get("analytic_break_even"),
        "cost_model": meta.get("cost_model", {}),
        "cost_inr": meta.get("cost_inr", {}),
        "currency": meta.get("currency", {}),
        "selection_rule": meta.get("selection_rule"),
        "rationale": meta.get("rationale"),
        "marginal_gain_threshold": meta.get("marginal_gain_threshold"),
        "relative_gain_over_logreg": meta.get("relative_gain_over_logreg"),
        "gbp_to_inr": cm.GBP_TO_INR,
        "data_is_synthetic": meta.get("DATA_IS_SYNTHETIC", False),
    })


@app.route("/api/score", methods=["POST"])
def score():
    """Score one order. Expects JSON with the 17 features."""
    if not MODEL_LOADED:
        return jsonify({"error": MODEL_ERROR}), 503

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Expected a JSON object of features"}), 400

    order = {}
    missing = []
    for f in scorer.features:
        if f in data:
            try:
                order[f] = float(data[f])
            except (ValueError, TypeError):
                return jsonify({"error": f"Feature '{f}' must be numeric"}), 400
        else:
            missing.append(f)

    if missing:
        return jsonify({"error": f"Missing features: {missing}"}), 400

    try:
        result = scorer.score_order(order, top_n=8)
    except (ValueError, KeyError) as e:
        # A NaN or an out-of-range value is the caller's problem, not the
        # server's: Scorer rejects a non-finite matrix rather than letting it
        # score as 0.5. That is a 400, not a 500.
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    result["gbp_to_inr"] = cm.GBP_TO_INR
    return jsonify(result)


@app.route("/api/sweep", methods=["GET"])
def sweep():
    """The pre-computed threshold sweep, shipped model only."""
    if not MODEL_LOADED:
        return jsonify({"error": MODEL_ERROR}), 503
    if not SWEEP_DATA:
        return jsonify({"error": SWEEP_ERROR
                        or "Threshold sweep data not available"}), 404

    data = [{**row, "total_cost_inr": cm.to_inr(row["total_cost_pence"])}
            for row in SWEEP_DATA]

    # Flagging nothing: no flag is reviewed, nobody is wrongly flagged, and
    # every positive is a miss. Computed here rather than in the browser so the
    # cost model stays on one side of the wire.
    cost_none_pence = cm.total_cost_from_counts(tp=0, fp=0, fn=TOTAL_POSITIVES)

    return jsonify({
        "sweep": data,
        "model": scorer.meta.get("winner"),
        "shipped_threshold": scorer.threshold,
        "cost_flag_nothing_pence": cost_none_pence,
        "cost_flag_nothing_inr": cm.to_inr(cost_none_pence),
        "gbp_to_inr": cm.GBP_TO_INR,
    })


def _cost_inputs(data):
    """
    Request body -> the three composite costs and the prevention rate.

    Every money field arrives in INR because that is what the page displays;
    the model is denominated in pence. Both the conversion and the arithmetic
    come from cost_model.py - this endpoint used to carry its own copy of both,
    which is how it ended up with a differently-guarded break-even from the one
    the Streamlit app showed for identical inputs.

    c_return_mult / c_review_mult scale a finished composite cost. The
    sensitivity table needs "what if a return cost twice as much", which is not
    the same question as "what if half as much were recovered" - the recovery
    rate only touches one of the three terms in c_return.
    """
    def money(key, default_pence):
        return cm.inr_to_pence(float(data.get(key, cm.to_inr(default_pence))))

    def rate(key, default):
        return float(data.get(key, default))

    prevention = rate("prevention", cm.PREVENTION_RATE)
    c_review, c_friction, c_return = cm.derive_costs(
        v_returned_pence=money("v_ret_inr", cm.RETURNED_ORDER_VALUE_PENCE),
        v_kept_pence=money("v_kept_inr", cm.KEPT_ORDER_VALUE_PENCE),
        recovery=rate("recovery", cm.GOODS_RECOVERY_RATE),
        logistics_pence=money("logistics_inr", cm.RETURN_LOGISTICS_PENCE),
        review_pence=money("review_inr", cm.COST_REVIEW_PENCE),
        abandon=rate("abandon", cm.ABANDON_RATE),
        margin=rate("margin", cm.CONTRIBUTION_MARGIN_RATE))

    c_return *= rate("c_return_mult", 1.0)
    c_review *= rate("c_review_mult", 1.0)
    return c_review, c_friction, c_return, prevention


@app.route("/api/cost-optimal", methods=["POST"])
def cost_optimal():
    """
    Re-cost the sweep under user-supplied cost assumptions.

    All money inputs are in INR; every one of them is optional and falls back
    to the shipped assumption, so an empty body returns the shipped curve.
    """
    if not MODEL_LOADED:
        return jsonify({"error": MODEL_ERROR}), 503
    if not SWEEP_DATA:
        return jsonify({"error": SWEEP_ERROR
                        or "Threshold sweep data not available"}), 404

    data = request.get_json(silent=True)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        return jsonify({"error": "Expected a JSON object of cost inputs"}), 400

    try:
        c_review, c_friction, c_return, prevention = _cost_inputs(data)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Bad cost input: {e}"}), 400

    p_star = cm.analytic_break_even(c_review, c_friction, c_return, prevention)

    kw = dict(c_review=c_review, c_friction=c_friction,
              c_return=c_return, prevention=prevention)
    recosted = [{"threshold": row["threshold"],
                 "total_cost_pence": cm.total_cost_from_counts(
                     row["tp"], row["fp"], row["fn"], row["tn"], **kw)}
                for row in SWEEP_DATA]
    for r in recosted:
        r["total_cost_inr"] = cm.to_inr(r["total_cost_pence"])

    best = min(recosted, key=lambda r: r["total_cost_pence"])
    cost_none_pence = cm.total_cost_from_counts(0, 0, TOTAL_POSITIVES, **kw)

    return jsonify({
        "p_star": round(p_star, 6),
        "c_return_pence": round(c_return, 2),
        "c_friction_pence": round(c_friction, 2),
        "c_review_pence": round(c_review, 2),
        "recosted_sweep": recosted,
        "best_threshold": best["threshold"],
        "best_cost_inr": round(best["total_cost_inr"], 2),
        "cost_flag_nothing_inr": round(cm.to_inr(cost_none_pence), 2),
        "shipped_threshold": scorer.threshold,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
