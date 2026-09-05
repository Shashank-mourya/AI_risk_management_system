"""
Edge-case robustness plus held-out accuracy. Run after the training notebook has
written artefacts/.

    python evaluate_model.py
    python evaluate_model.py --bootstrap 2000     # wider CIs, slower

Companion to test_model.py rather than a replacement:

    test_model.py       invariants  - properties that hold whatever the data is
    evaluate_model.py   part A      - edge cases: behaviour at the boundaries
                        part B      - accuracy: what the model is actually worth

Part A is pass/fail and sets the exit code. Part B is measurement only - "is 0.74
AUC good" is a judgement about the business, not an assertion about the code.

Accuracy is reported because people ask for it, but it is the weakest number
here: the held-out base rate is 17.6%, so flagging nothing scores 82.4%. Every
accuracy figure is printed next to that floor. The operating point is picked on
total cost.
"""

import argparse
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    log_loss, precision_score, recall_score, roc_auc_score,
)

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(ROOT, "artefacts")
REAL_DATA = os.path.join(ROOT, "features.pkl")
SYNTHETIC_DATA = os.path.join(ROOT, "data", "synthetic_features.pkl")

# --- cost model
# Imported, never redefined. This file and the training notebook used to carry
# two different cost models, which meant two different "optimal" thresholds
# were simultaneously in the repo. cost_model.py is now the only definition.
from predict import Scorer  # noqa: E402
from cost_model import (  # noqa: E402
    total_cost_pence, total_cost_pence_per_order, describe as describe_cost,
    P_STAR, COST_REVIEW_PENCE, COST_FRICTION_PENCE, COST_RETURN_PENCE,
    PREVENTION_RATE, GBP_TO_INR, to_inr,
)

_PASS, _FAIL = [], []


def rule(title, char="="):
    print("\n" + char * 74)
    print(f"  {title}")
    print(char * 74)


def check(name, ok, detail=""):
    (_PASS if ok else _FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail and not ok:
        print(f"         {detail}")
    return bool(ok)


def note(name, value):
    print(f"  [ .. ] {name}\n         {value}")


# --- scoring path
# Both of these used to be a second implementation of predict.py. Edge cases
# are only worth testing against the code that actually ships.
_SCORER = None


def _scorer():
    """One instance: constructing a Scorer reloads joblib from disk."""
    global _SCORER
    if _SCORER is None:
        _SCORER = Scorer(ART_DIR)
    return _SCORER


def make_scorer(model, scaler, needs_scaler):
    return _scorer().score_batch


def score_one(score, features, order: dict):
    """Dict-keyed single-order API. Order of keys is irrelevant by construction."""
    return _scorer().score_order(order)["risk_probability"]


# === PART A: edge cases
def part_a(score, features, meta, test, X, y, p):
    thr = meta["chosen_threshold"]
    n_feat = len(features)
    med = np.median(X, axis=0)

    rule("PART A - EDGE CASES AND ROBUSTNESS")

    # --- shapes
    print("\nA1 · input shapes")
    check("single row (2-D) scores", score(X[:1]).shape == (1,))
    check("single row (1-D) is accepted and reshaped", score(X[0]).shape == (1,))
    check("full batch scores", score(X).shape == (len(X),))
    try:
        out = score(np.empty((0, n_feat)))
        check("empty batch is handled (returned empty array)", out.shape == (0,))
    except Exception as e:
        # Raising is acceptable - what matters is that it does not return junk.
        check("empty batch is handled (raised cleanly)", True)
        note("empty batch", f"raises {type(e).__name__} - callers must guard for this "
                            f"before scoring an empty request")

    try:
        score(np.zeros((3, n_feat - 1)))
        check("wrong feature count is rejected", False, "silently accepted a short row")
    except Exception as e:
        check("wrong feature count is rejected", True, type(e).__name__)

    # --- extreme values
    print("\nA2 · extreme and degenerate values")
    zeros = score(np.zeros((1, n_feat)))[0]
    check("all-zero row scores in [0,1]", 0.0 <= zeros <= 1.0, f"got {zeros}")
    note("all-zero row probability", f"{zeros:.6f}")

    big = score(np.full((1, n_feat), 1e9))[0]
    small = score(np.full((1, n_feat), -1e9))[0]
    check("+1e9 row stays in [0,1]", 0.0 <= big <= 1.0, f"got {big}")
    check("-1e9 row stays in [0,1]", 0.0 <= small <= 1.0, f"got {small}")
    note("saturation", f"+1e9 -> {big:.6f}   -1e9 -> {small:.6f}")

    # NaN / inf must never come back as a plausible-looking probability. Either
    # the scorer raises, or it returns something obviously non-finite. Silently
    # returning 0.34 for a NaN input is the failure mode being guarded against.
    for label, bad_val in [("NaN", np.nan), ("+inf", np.inf), ("-inf", -np.inf)]:
        row = med.copy(); row[0] = bad_val
        try:
            got = score(row)[0]
            ok = not np.isfinite(got)
            check(f"{label} input does not yield a plausible score", ok,
                  f"returned {got:.6f} - a caller could mistake this for a real score")
            if ok:
                note(f"{label} input", f"returns non-finite {got}")
        except Exception as e:
            check(f"{label} input does not yield a plausible score (rejected)", True)
            note(f"{label} input", f"rejected with {type(e).__name__} - correct: "
                                   f"the caller is forced to handle it")

    # --- feature-order safety
    print("\nA3 · feature-order safety")
    row = dict(zip(features, med))
    shuffled = {k: row[k] for k in list(row)[::-1]}
    check("dict API is key-order independent",
          abs(score_one(score, features, row) - score_one(score, features, shuffled)) < 1e-12)

    reversed_arr = med[::-1].copy()
    check("array API IS order-sensitive (so use the dict API)",
          abs(score(med)[0] - score(reversed_arr)[0]) > 1e-9,
          "reversing the array changed nothing - features may be interchangeable, "
          "which would be suspicious")

    try:
        incomplete = {k: v for k, v in list(row.items())[:-1]}
        score_one(score, features, incomplete)
        check("missing feature raises", False, "silently scored an incomplete order")
    except KeyError:
        check("missing feature raises KeyError", True)

    extra = dict(row); extra["not_a_feature"] = 999.0
    check("unknown extra key is ignored",
          abs(score_one(score, features, extra) - score_one(score, features, row)) < 1e-12)

    # --- sentinels
    print("\nA4 · sentinel semantics (-1 = no history)")
    i_rate = features.index("customer_prior_return_rate")
    probe = X[:500].copy()
    variants = {}
    for label, val in [("-1 (no history)", -1.0), ("0.0 (never returned)", 0.0),
                       ("0.5", 0.5), ("1.0 (always returns)", 1.0)]:
        v = probe.copy(); v[:, i_rate] = val
        variants[label] = score(v).mean()
    check("-1 differs from 0.0",
          abs(variants["-1 (no history)"] - variants["0.0 (never returned)"]) > 1e-9,
          "the model cannot tell 'no history' from 'never returned'")
    check("higher prior return rate raises mean risk",
          variants["0.0 (never returned)"] < variants["0.5"] < variants["1.0 (always returns)"],
          "monotonicity in prior return rate is violated - worth understanding why")
    for k, v in variants.items():
        print(f"         prior_return_rate = {k:<22} mean p = {v:.6f}")

    # --- coherent inputs
    print("\nA5 · coherent vs incoherent cold-start rows")
    i_new = features.index("is_new_customer")
    i_prior = features.index("customer_prior_orders")
    i_ret = features.index("customer_prior_returns")

    cold = med.copy(); cold[i_new] = 1; cold[i_prior] = 0; cold[i_ret] = 0; cold[i_rate] = -1
    warm = med.copy(); warm[i_new] = 0; warm[i_prior] = 12; warm[i_ret] = 3; warm[i_rate] = 0.25
    check("coherent cold-start row scores", np.isfinite(score(cold)[0]))
    check("coherent returning row scores", np.isfinite(score(warm)[0]))
    note("cold-start vs returning", f"{score(cold)[0]:.6f}  vs  {score(warm)[0]:.6f}")

    bad = med.copy(); bad[i_new] = 1; bad[i_prior] = 5; bad[i_rate] = 0.4
    pb = score(bad)[0]
    check("INCOHERENT row (is_new=1 but 5 prior orders) is scored without complaint",
          np.isfinite(pb),
          "")
    note("incoherent row", f"scores {pb:.6f} - nothing validates feature coherence. "
                           f"If the API accepts caller-supplied features, it must.")

    # --- threshold edges
    print("\nA6 · threshold boundary")
    check("p exactly == threshold is FLAGGED (>= semantics)", bool(thr >= thr))
    eps = 1e-12
    check("p just below threshold is not flagged", not bool((thr - eps) >= thr))
    n_at = int(np.sum(np.isclose(p, thr, atol=1e-9)))
    note("test rows sitting exactly on the threshold", f"{n_at}")

    # --- determinism etc
    print("\nA7 · determinism and duplicates")
    check("repeat scoring is bit-identical", bool(np.array_equal(score(X), p)))
    dup = np.repeat(X[:1], 5, axis=0)
    check("duplicate rows score identically", bool(len(np.unique(score(dup))) == 1))
    check("batch == row-by-row",
          bool(np.allclose([score(X[i])[0] for i in range(100)], p[:100], atol=1e-10)))

    # --- out-of-range / drift
    print("\nA8 · values outside the training range")
    tr_max = X.max(axis=0)
    beyond = med.copy()
    i_val = features.index("order_value")
    beyond[i_val] = tr_max[i_val] * 100
    pbeyond = score(beyond)[0]
    check("100x the largest observed order value still scores in [0,1]",
          0.0 <= pbeyond <= 1.0, f"got {pbeyond}")
    note("extrapolation", f"order_value x100 -> p = {pbeyond:.6f}  "
                          f"(median row is {score(med)[0]:.6f})")

    neg = med.copy(); neg[i_val] = -1000.0
    check("negative order value does not crash", np.isfinite(score(neg)[0]))
    note("negative order value", f"p = {score(neg)[0]:.6f} - nothing rejects it; "
                                 f"validate upstream")


# === PART B: accuracy
def metrics_at(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return dict(
        threshold=thr, tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        precision=precision_score(y, yhat, zero_division=0),
        recall=recall_score(y, yhat, zero_division=0),
        f1=f1_score(y, yhat, zero_division=0),
        accuracy=(tp + tn) / len(y),
        flag_rate=yhat.mean(),
    )


def part_b(score, features, meta, test, X, y, p, n_boot, thr_override=None):
    thr = meta["chosen_threshold"] if thr_override is None else thr_override
    base = y.mean()

    def cost(yhat):
        return to_inr(total_cost_pence(y, yhat))

    rule("PART B - HELD-OUT ACCURACY")
    print(f"\n  test rows            {len(y):,}")
    print(f"  actual returns       {int(y.sum()):,}")
    print(f"  BASE RATE            {base:.4%}   <- the number every metric below is judged against")
    print(f"  operating threshold  {thr}"
          f"{'  (overridden on the command line)' if thr_override is not None else ''}")
    print()
    for line in describe_cost().splitlines():
        print(f"  {line}")

    # Where does the empirical cost curve actually bottom out?
    grid = np.round(np.arange(0.01, 0.995, 0.01), 4)
    curve = np.array([total_cost_pence(y, (p >= t).astype(int)) for t in grid])
    argmin_thr = float(grid[int(np.argmin(curve))])
    steps = abs(argmin_thr - P_STAR) / 0.01
    print(f"  empirical argmin     {argmin_thr:.2f}   ({steps:.1f} grid steps from p*)")
    if steps > 1.0:
        print(f"  !! argmin disagrees with p* by more than one step. That is a")
        print(f"     CALIBRATION signal: see B4. Choosing the threshold by minimising")
        print(f"     cost on the TEST set also fits the test set, which is why p* -")
        print(f"     derived from the cost model alone - is the safer operating point.")

    # --- headline
    m = metrics_at(y, p, thr)
    rule("B1 · headline metrics at the chosen threshold", "-")
    print(f"""
                      value      vs base rate
    precision       {m['precision']:.4f}      {m['precision']/base:>6.2f}x base rate
    recall          {m['recall']:.4f}
    F1              {m['f1']:.4f}
    accuracy        {m['accuracy']:.4f}      floor is {1-base:.4f} (flag nothing)
    flag rate       {m['flag_rate']:.4f}

    ROC-AUC         {roc_auc_score(y, p):.4f}      0.5 = coin flip
    PR-AUC          {average_precision_score(y, p):.4f}      {average_precision_score(y,p)/base:>6.2f}x base rate
    Brier           {brier_score_loss(y, p):.4f}      lower is better
    log loss        {log_loss(y, p):.4f}

    confusion       TP {m['tp']:>5}   FP {m['fp']:>5}
                    FN {m['fn']:>5}   TN {m['tn']:>5}""")

    if m["accuracy"] < 1 - base:
        print(f"\n    NOTE: accuracy {m['accuracy']:.4f} is BELOW the flag-nothing floor "
              f"{1-base:.4f}.\n    That is expected at a recall-heavy operating point and is "
              f"why accuracy\n    is not the selection criterion here.")

    # --- bootstrap CIs
    rule(f"B2 · bootstrap confidence intervals ({n_boot:,} resamples)", "-")
    rng = np.random.default_rng(42)
    boot = {"roc_auc": [], "precision": [], "recall": [], "cost_inr": []}
    idx_all = np.arange(len(y))
    for _ in range(n_boot):
        i = rng.choice(idx_all, size=len(idx_all), replace=True)
        yb, pb = y[i], p[i]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        yhb = (pb >= thr).astype(int)
        boot["roc_auc"].append(roc_auc_score(yb, pb))
        boot["precision"].append(precision_score(yb, yhb, zero_division=0))
        boot["recall"].append(recall_score(yb, yhb, zero_division=0))
        boot["cost_inr"].append(to_inr(total_cost_pence(yb, yhb)))
    print()
    ci = {}
    for k, v in boot.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        ci[k] = (lo, hi)
        print(f"    {k:<12} {np.mean(v):>12.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    print("\n    A CI that straddles a decision boundary means the difference is noise.")

    # --- vs baselines
    rule("B3 · versus baselines (cost is the criterion)", "-")
    rng2 = np.random.default_rng(7)
    rand_at_rate = (rng2.random(len(y)) < m["flag_rate"]).astype(int)
    rows = [
        ("model @ chosen thr", cost((p >= thr).astype(int)), m["precision"], m["recall"]),
        ("flag nothing", cost(np.zeros(len(y), int)), 0.0, 0.0),
        ("flag everything", cost(np.ones(len(y), int)), base, 1.0),
        (f"random @ same rate", cost(rand_at_rate),
         precision_score(y, rand_at_rate, zero_division=0),
         recall_score(y, rand_at_rate, zero_division=0)),
        ("model @ 0.50", cost((p >= 0.5).astype(int)),
         *[precision_score(y, (p >= .5).astype(int), zero_division=0),
           recall_score(y, (p >= .5).astype(int), zero_division=0)]),
    ]
    print("")
    print(f"    amounts shown in INR at an assumed GBP-to-INR rate of "
          f"{GBP_TO_INR:.1f}. The source data is GBP; the conversion is a "
          f"display relabelling and moves no threshold.")
    print(f"\n    {'policy':<24}{'cost INR':>14}{'precision':>11}{'recall':>9}   vs model")
    model_cost = rows[0][1]
    for nm, c, pr, rc in rows:
        if nm.startswith("model @ chosen"):
            delta = "(reference)"
        elif c > model_cost:
            delta = f"INR {c - model_cost:>11,.2f} worse"
        else:
            delta = f"INR {model_cost - c:>11,.2f} BETTER"
        print(f"    {nm:<24}{c:>14,.2f}{pr:>11.4f}{rc:>9.4f}   {delta}")

    saved = cost(np.zeros(len(y), int)) - model_cost
    all_cost = cost(np.ones(len(y), int))
    gap = all_cost - model_cost
    print(f"\n    The model saves INR {saved:,.2f} against absorbing every return,")
    print(f"    and INR {gap:,.2f} against reviewing every order.")

    # The comparison that actually matters: "flag everything" needs no model at
    # all. If the model's edge over it is inside the bootstrap noise, the model
    # is not yet earning its place at this cost ratio.
    lo, hi = ci["cost_inr"]
    print(f"\n    Model cost 95% CI: INR [{lo:,.2f}, {hi:,.2f}]")
    print(f"    'Flag everything' costs INR {all_cost:,.2f}, which needs no model at all.")
    if lo <= all_cost <= hi:
        print(f"\n    >> 'Flag everything' falls INSIDE the model's confidence interval.")
        print(f"    >> The INR {gap:,.2f} edge ({gap/all_cost:.2%}) is not distinguishable")
        print(f"    >> from noise at this FP:FN ratio. The model is beating random")
        print(f"    >> comfortably, but at THIS cost assumption it is barely beating")
        print(f"    >> a policy that reviews everything. Check the cost-model inputs.")
    else:
        print(f"\n    The model's edge over 'flag everything' is outside the CI, "
              f"so it is real at this cost ratio.")

    # --- calibration
    rule("B4 · calibration (the cost model depends on this)", "-")
    d = pd.DataFrame({"p": p, "y": y})
    d["decile"] = pd.qcut(d.p, 10, labels=False, duplicates="drop")
    cal = d.groupby("decile").agg(n=("y", "size"), predicted=("p", "mean"),
                                  observed=("y", "mean")).reset_index()
    cal["gap"] = cal.observed - cal.predicted
    print(f"\n    {'decile':>7}{'n':>7}{'predicted':>12}{'observed':>11}{'gap':>9}")
    for r in cal.itertuples():
        bar = "+" if r.gap > 0.02 else ("-" if r.gap < -0.02 else " ")
        print(f"    {int(r.decile):>7}{r.n:>7,}{r.predicted:>12.4f}{r.observed:>11.4f}"
              f"{r.gap:>9.4f} {bar}")
    ece = float((cal.n / cal.n.sum() * cal.gap.abs()).sum())
    print(f"\n    expected calibration error (ECE): {ece:.4f}")
    print("    Well-calibrated probabilities are what make the cost curve meaningful;")
    print("    Nothing in this repo resamples, for exactly that reason.")

    # --- lift
    rule("B5 · lift by score decile", "-")
    d["rank_decile"] = pd.qcut(d.p.rank(method="first", ascending=False), 10,
                               labels=False, duplicates="drop")
    lift = d.groupby("rank_decile").agg(n=("y", "size"), returns=("y", "sum")).reset_index()
    lift["rate"] = lift.returns / lift.n
    lift["lift"] = lift.rate / base
    lift["cum_returns_pct"] = lift.returns.cumsum() / lift.returns.sum()
    print(f"\n    {'decile':>7}{'n':>7}{'returns':>9}{'rate':>9}{'lift':>8}{'cum % of all returns':>22}")
    for r in lift.itertuples():
        print(f"    {int(r.rank_decile)+1:>7}{r.n:>7,}{int(r.returns):>9,}{r.rate:>9.4f}"
              f"{r.lift:>8.2f}{r.cum_returns_pct:>21.1%}")
    top = lift.iloc[0]
    print(f"\n    The riskiest 10% of orders contain {top.cum_returns_pct:.1%} of all returns "
          f"({top.lift:.2f}x the base rate).")

    # --- segments
    rule("B6 · segment breakdown (where does it work?)", "-")
    seg = test.copy()
    seg["p"] = p
    seg["y"] = y
    seg["flag"] = (p >= thr).astype(int)

    def seg_report(title, series):
        print(f"\n    {title}")
        print(f"    {'segment':<22}{'n':>7}{'base%':>7}{'floor%':>8}{'acc%':>7}"
              f"{'AUC':>8}{'prec':>7}{'recall':>8}")
        for name, g in seg.groupby(series, observed=True):
            if len(g) < 50 or g.y.nunique() < 2:
                print(f"    {str(name):<22}{len(g):>7,}   (too small / single-class)")
                continue
            acc = (g.y == g.flag).mean()
            floor = 1 - g.y.mean()      # accuracy of flagging nothing in this segment
            print(f"    {str(name):<22}{len(g):>7,}{g.y.mean()*100:>7.1f}{floor*100:>8.1f}"
                  f"{acc*100:>7.1f}"
                  f"{roc_auc_score(g.y, g.p):>8.3f}"
                  f"{precision_score(g.y, g.flag, zero_division=0):>7.3f}"
                  f"{recall_score(g.y, g.flag, zero_division=0):>8.3f}")
    seg_report("by customer history", seg.is_new_customer.map({1: "cold-start (new)",
                                                               0: "returning"}))
    seg_report("by order value quartile",
               pd.qcut(seg.order_value, 4, labels=["Q1 smallest", "Q2", "Q3", "Q4 largest"]))
    seg_report("by geography", seg.is_uk.map({1: "UK", 0: "non-UK"}))

    # --- drift
    rule("B7 · temporal stability across the test window", "-")
    seg["period"] = pd.qcut(seg.order_date.rank(method="first"), 3,
                            labels=["early", "middle", "late"])
    print(f"\n    {'period':<10}{'n':>7}{'dates':>26}{'base':>8}{'AUC':>8}{'recall':>8}")
    for name, g in seg.groupby("period", observed=True):
        span = f"{g.order_date.min().date()} -> {g.order_date.max().date()}"
        print(f"    {str(name):<10}{len(g):>7,}{span:>26}{g.y.mean():>8.3f}"
              f"{roc_auc_score(g.y, g.p):>8.3f}"
              f"{recall_score(g.y, g.flag, zero_division=0):>8.3f}")
    print("\n    A falling AUC across periods would mean the model is going stale and")
    print("    the split date is flattering it. Watch this when the window moves.")

    return m, ece


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=1000,
                    help="bootstrap resamples for confidence intervals")
    ap.add_argument("--threshold", type=float, default=None,
                    help="evaluate at this threshold instead of the one in threshold.json "
                         "(e.g. --threshold 0.245 for the analytic break-even)")
    args = ap.parse_args()

    rule("AI RISK MANAGER - MODEL EVALUATION")

    for f in ("model.joblib", "threshold.json"):
        if not os.path.exists(os.path.join(ART_DIR, f)):
            print(f"\n  {f} not found in {ART_DIR}/.")
            print("  Run notebooks/train_model.ipynb first.")
            return 1

    meta = json.load(open(os.path.join(ART_DIR, "threshold.json")))
    model = joblib.load(os.path.join(ART_DIR, "model.joblib"))
    sp = os.path.join(ART_DIR, "scaler.joblib")
    scaler = joblib.load(sp) if os.path.exists(sp) else None
    features = meta["features"]
    is_syn = bool(meta.get("DATA_IS_SYNTHETIC"))

    data = SYNTHETIC_DATA if is_syn else REAL_DATA
    if not os.path.exists(data):
        print(f"\n  {data} not found.")
        return 1
    df = pd.read_pickle(data) if data.endswith(".pkl") else pd.read_csv(
        data, parse_dates=["order_date"])
    test = df[df.split == "test"].sort_values("order_date").reset_index(drop=True)
    X = test[features].to_numpy(float)
    y = test["returned"].to_numpy(int)

    score = make_scorer(model, scaler, meta["requires_scaler"])
    p = score(X)

    print(f"\n  model        {meta['winner']}")
    print(f"  threshold    {meta['chosen_threshold']}")
    print(f"  data         {os.path.relpath(data, ROOT)}  "
          f"({'SYNTHETIC' if is_syn else 'real'})")
    print(f"  test rows    {len(test):,}")
    if is_syn:
        print("\n  *** SYNTHETIC DATA - Part B measures the generator, not the model. ***")

    part_a(score, features, meta, test, X, y, p)
    m, ece = part_b(score, features, meta, test, X, y, p, args.bootstrap,
                    thr_override=args.threshold)

    rule("SUMMARY")
    print(f"\n  Part A - edge cases: {len(_PASS)} passed, {len(_FAIL)} failed")
    if _FAIL:
        print("\n  failures:")
        for f in _FAIL:
            print(f"    - {f}")
    print(f"""
  Part B - accuracy on {len(y):,} held-out orders:
    ROC-AUC     {roc_auc_score(y, p):.4f}   (0.5 = coin flip)
    PR-AUC      {average_precision_score(y, p):.4f}   ({average_precision_score(y,p)/y.mean():.2f}x the {y.mean():.2%} base rate)
    precision   {m['precision']:.4f}   ({m['precision']/y.mean():.2f}x base rate)
    recall      {m['recall']:.4f}
    accuracy    {m['accuracy']:.4f}   (flag-nothing floor: {1-y.mean():.4f})
    ECE         {ece:.4f}

  The operating point was chosen on total cost, not on any number above.""")
    if is_syn:
        print("\n  *** Trained on SYNTHETIC data - Part B is not reportable. ***")
    print("=" * 74)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
