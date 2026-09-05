"""
Acceptance suite for the data pipeline. Run after build_labels.py,
build_features.py and build_database.py.

    python test_phase1.py

Scope is a feature matrix that is reproducible, free of target leakage and
consistent with the database. Model quality belongs to test_model.py and
evaluate_model.py.

  1  artefact contract    - the six generated files exist and are readable
  2  verified numbers     - the recorded counts reproduce exactly
  3  split integrity      - chronological, disjoint, correct cut date
  4  maturity             - no order inside 90 days of the data end survives,
                          and the positive rate does not drift with how long
                          each order happened to be watched
  5  label sanity         - binary, matches orders_labeled, no NaN
  6  sentinel semantics   - -1 means no history and tracks is_new_customer
  7  leakage: columns     - no outcome or identifier column is a feature
  8  leakage: as-of       - prior returns re-derived independently and compared
  9  database             - FK violations, row counts, empty-by-design tables
 10  reproducibility      - features.pkl and features.csv agree
 11  gitignore            - generated files ignored, .env.example tracked

Check 8 is the one that matters. It rebuilds customer_prior_returns from
retail2.pkl with a slow explicit loop over return dates and demands an exact
match against the vectorised searchsorted implementation in build_features.py.
Two independent implementations agreeing is the only real evidence the as-of
rule holds.
"""

import os
import re
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd

from config import (  # noqa: E402
    FEATURES, MIN_GAP_DAYS, RETURN_WINDOW_DAYS, genuine_returns,
)

ROOT = os.path.dirname(os.path.abspath(__file__))

# From BUILD_PLAN.md. These are measured, not guessed. A mismatch means
# something real changed upstream and must be investigated, not patched over.
EXPECTED = {
    "rows_source":      1_067_371,
    "mature":              30_347,
    "immature":             6_628,
    "n_train":             24_277,
    "n_test":               6_070,
    "positive_rate":       0.1673,
    "train_positive":      0.1683,
    "test_positive":       0.1633,
    "split_date":     "2011-04-28",
    "credit_lines":        18_744,
    "nonc_negatives":       3_457,
}

_PASS, _FAIL = [], []


def check(name, condition, detail=""):
    (_PASS if condition else _FAIL).append(name)
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"\n         {detail}" if detail and not condition else ""))
    return bool(condition)


def near(a, b, tol=5e-5):
    return abs(float(a) - float(b)) <= tol


def main():
    print("=" * 72)
    print("  Phase 1 acceptance suite - data pipeline")
    print("=" * 72)

    # --- 1 artefact contract
    print("\n1 - artefact contract")
    paths = {
        "retail2.pkl":         os.path.join(ROOT, "retail2.pkl"),
        "orders_labeled.pkl":  os.path.join(ROOT, "orders_labeled.pkl"),
        "orders_labeled.csv":  os.path.join(ROOT, "orders_labeled.csv"),
        "features.pkl":        os.path.join(ROOT, "features.pkl"),
        "features.csv":        os.path.join(ROOT, "features.csv"),
        "risk.db":             os.path.join(ROOT, "risk.db"),
    }
    for name, p in paths.items():
        if not check(f"{name} exists", os.path.exists(p)):
            print("\nCannot continue. Run: python build_labels.py && "
                  "python build_features.py && python build_database.py")
            return 1

    df = pd.read_pickle(paths["retail2.pkl"])
    orders = pd.read_pickle(paths["orders_labeled.pkl"])
    feat = pd.read_pickle(paths["features.pkl"])

    # --- 2 verified numbers
    print("\n2 - verified numbers reproduce (BUILD_PLAN.md)")
    check(f"source rows = {EXPECTED['rows_source']:,}",
          len(df) == EXPECTED["rows_source"], f"got {len(df):,}")
    check(f"mature orders = {EXPECTED['mature']:,}",
          len(orders) == EXPECTED["mature"], f"got {len(orders):,}")
    check(f"feature rows = {EXPECTED['mature']:,}",
          len(feat) == EXPECTED["mature"],
          f"got {len(feat):,} - dropna on features removed rows")

    pos = feat.returned.mean()
    check(f"positive rate = {EXPECTED['positive_rate']:.2%}",
          near(pos, EXPECTED["positive_rate"]), f"got {pos:.4%}")

    tr = feat[feat.split == "train"]
    te = feat[feat.split == "test"]
    check(f"train rows = {EXPECTED['n_train']:,}",
          len(tr) == EXPECTED["n_train"], f"got {len(tr):,}")
    check(f"test rows = {EXPECTED['n_test']:,}",
          len(te) == EXPECTED["n_test"], f"got {len(te):,}")
    check(f"train positive = {EXPECTED['train_positive']:.2%}",
          near(tr.returned.mean(), EXPECTED["train_positive"]),
          f"got {tr.returned.mean():.4%}")
    check(f"test positive = {EXPECTED['test_positive']:.2%}",
          near(te.returned.mean(), EXPECTED["test_positive"]),
          f"got {te.returned.mean():.4%}")

    # build_labels.py counts only credit lines with a usable CustomerID, since a
    # return that cannot be attributed to a customer cannot become a label.
    credit = df[df.isC & (df.Quantity < 0) & df.CustomerID.notna()]
    check(f"credit-note lines (customer known) = {EXPECTED['credit_lines']:,}",
          len(credit) == EXPECTED["credit_lines"], f"got {len(credit):,}")
    nonc = df[(~df.isC) & (df.Quantity < 0)]
    check(f"non-credit negatives = {EXPECTED['nonc_negatives']:,} (excluded)",
          len(nonc) == EXPECTED["nonc_negatives"], f"got {len(nonc):,}")
    check("non-credit negatives are not in the order set",
          not set(nonc.Invoice.astype(str)) & set(orders.Invoice.astype(str)))

    # --- 3 split integrity
    print("\n3 - split integrity (chronological only)")
    split_date = pd.Timestamp(EXPECTED["split_date"])
    check(f"split date = {EXPECTED['split_date']}",
          te.order_date.min().normalize() == split_date,
          f"first test order {te.order_date.min()}")
    check("every train order precedes every test order",
          tr.order_date.max() < te.order_date.min(),
          f"train max {tr.order_date.max()} vs test min {te.order_date.min()}")
    check("no invoice appears in both splits",
          not (set(tr.Invoice) & set(te.Invoice)))
    check("split labels are exactly {train, test}",
          set(feat.split.unique()) == {"train", "test"},
          f"got {sorted(feat.split.unique())}")
    check("train + test = all rows", len(tr) + len(te) == len(feat))

    # --- 4 maturity
    print("\n4 - maturity (90-day window)")
    data_end = df.InvoiceDate.max()
    cutoff = data_end - pd.Timedelta(days=RETURN_WINDOW_DAYS)
    check(f"no order after the maturity cutoff {cutoff.date()}",
          feat.order_date.max() <= cutoff,
          f"latest order {feat.order_date.max()}")
    print(f"         data end {data_end.date()}  cutoff {cutoff.date()}")
    print(f"         mature {len(feat):,}  (immature excluded upstream: "
          f"{EXPECTED['immature']:,})")

    # the CHECK that would have caught the label-horizon bug.
    #
    # Maturity guarantees every order had RETURN_WINDOW_DAYS to be observed. If
    # the label counts returns beyond that window, then an order watched for two
    # years is likelier to be labelled positive than one watched for ninety days
    # - for reasons that have nothing to do with the order. The positive rate
    # then encodes exposure time, and because the split is chronological, train
    # sits in the long-window end and test in the short one, manufacturing a
    # train/test gap out of nothing.
    #
    # So: bucket orders by how long they were actually watched and require the
    # positive rate to be flat. It used to run 17.6% -> 20.3% across these
    # buckets; capping the label at the maturity window brought it into line.
    obs = (data_end - feat.order_date).dt.total_seconds() / 86400
    buckets = pd.qcut(obs, 5, labels=False, duplicates="drop")
    rates = feat.groupby(buckets, observed=True).returned.mean()
    spread = float(rates.max() - rates.min())
    # Tolerance calibrated so it actually catches the bug it exists for: the
    # uncapped label measured a spread of 0.027 (0.176 -> 0.203), the capped one
    # measures 0.012. A threshold of 0.03 would have passed the broken label.
    check("positive rate does not drift with observation-window length",
          spread < 0.02,
          f"spread {spread:.3f} across window quintiles: "
          f"{[round(r, 3) for r in rates.tolist()]} - the label horizon and the "
          f"maturity horizon have diverged")
    print(f"         positive rate by window quintile "
          f"{[round(r, 3) for r in rates.tolist()]}  spread {spread:.3f}")


    # --- 5 label sanity
    print("\n5 - label sanity")
    check("returned is binary", set(feat.returned.unique()) <= {0, 1},
          f"got {sorted(feat.returned.unique())}")
    check("no NaN in the label", not feat.returned.isna().any())
    nan_cols = feat[FEATURES].columns[feat[FEATURES].isna().any()].tolist()
    check("no NaN anywhere in the feature matrix", not nan_cols,
          f"cols with NaN: {nan_cols}")
    check("all feature values finite",
          bool(np.isfinite(feat[FEATURES].to_numpy(float)).all()))
    merged = feat[["Invoice", "returned"]].merge(
        orders[["Invoice", "returned"]], on="Invoice", suffixes=("_f", "_o"))
    check("labels agree with orders_labeled.pkl",
          bool((merged.returned_f == merged.returned_o).all()),
          f"{int((merged.returned_f != merged.returned_o).sum())} disagree")

    # --- 6 sentinel semantics
    print("\n6 - sentinel semantics (-1 is 'no history', not 'never returned')")
    rate = feat.customer_prior_return_rate
    n_sent = int((rate == -1).sum())
    check("the -1 sentinel survives into features.pkl", n_sent > 0)
    check("sentinel rows are exactly the new-customer rows",
          bool(((rate == -1) == (feat.is_new_customer == 1)).all()),
          f"sentinel {n_sent} vs new {int(feat.is_new_customer.sum())}")
    check("no rate is between -1 and 0 (nothing half-imputed)",
          not bool(((rate > -1) & (rate < 0)).any()))
    check("non-sentinel rates are non-negative",
          bool((rate[rate != -1] >= 0).all()))
    check("new customers have zero prior orders",
          bool((feat.loc[feat.is_new_customer == 1, "customer_prior_orders"] == 0).all()))
    print(f"         cold-start {n_sent:,} rows ({n_sent/len(feat):.1%})")

    # --- 7 leakage: columns
    print("\n7 - leakage: no outcome column is a feature")
    banned = {"returned", "return_date", "is_mature", "split", "Invoice",
              "customer_id", "label_risk", "return_gap_days"}
    check("no banned column in the feature list", not (banned & set(FEATURES)),
          f"leaky: {sorted(banned & set(FEATURES))}")
    check("feature list has exactly 17 entries", len(FEATURES) == 17)
    check("features.pkl carries every declared feature",
          not (set(FEATURES) - set(feat.columns)),
          f"missing: {sorted(set(FEATURES) - set(feat.columns))}")
    check("prior_returns never exceeds prior_orders",
          bool((feat.customer_prior_returns <= feat.customer_prior_orders).all()),
          f"{int((feat.customer_prior_returns > feat.customer_prior_orders).sum())} rows violate")
    # A feature that perfectly separates the label would be a smoking gun.
    worst = feat[FEATURES + ["returned"]].corr()["returned"].drop("returned").abs().max()
    check("no feature correlates > 0.95 with the label",
          worst < 0.95, f"max |corr| = {worst:.3f}")
    print(f"         strongest single-feature correlation {worst:.3f}")

    # --- 8 leakage: as-of
    print("\n8 - leakage: prior returns RE-DERIVED independently")
    purchases = df[(~df.isC) & (df.Quantity > 0) & df.CustomerID.notna()].copy()
    returns = df[df.isC & (df.Quantity < 0) & df.CustomerID.notna()].copy()
    purchases["pidx"] = purchases.index
    returns["ridx"] = returns.index
    pairs = returns[["ridx", "CustomerID", "StockCode", "InvoiceDate"]].merge(
        purchases[["pidx", "CustomerID", "StockCode", "InvoiceDate", "Invoice"]],
        on=["CustomerID", "StockCode"], suffixes=("_r", "_p"))
    pairs = pairs[pairs.InvoiceDate_p < pairs.InvoiceDate_r]
    pairs = pairs.sort_values("InvoiceDate_p").groupby("ridx", as_index=False).last()
    pairs["gap_days"] = (pairs.InvoiceDate_r - pairs.InvoiceDate_p).dt.total_seconds() / 86400
    pairs = genuine_returns(pairs)
    print(f"         rebuilt {len(pairs):,} purchase->return matches "
          f"in the [{MIN_GAP_DAYS:.0f}, {RETURN_WINDOW_DAYS}]d window")

    # Collapse to one event per returned purchase order, dated at its earliest
    # returned line. build_features.py counts prior returns in the same units
    # as prior orders; re-derive that here independently rather than trusting
    # it. Restricted to the labelled order population, same as the pipeline.
    order_return_date = pairs.groupby("Invoice").InvoiceDate_r.min()
    obs = (feat[["Invoice", "customer_id"]]
           .merge(order_return_date.rename("return_observed_at"),
                  left_on="Invoice", right_index=True, how="inner"))
    print(f"         collapsed to {len(obs):,} returned orders "
          f"across {obs.customer_id.nunique():,} customers")

    # Explicit loop: for each order, count this customer's returns whose return
    # date is strictly before the order date. Deliberately not searchsorted.
    ret_lists = {c: sorted(g.tolist())
                 for c, g in obs.groupby("customer_id").return_observed_at}
    expected = np.empty(len(feat), dtype=int)
    f_sorted = feat.sort_values("order_date").reset_index(drop=True)
    for i, (cust, when) in enumerate(zip(f_sorted.customer_id.values,
                                         f_sorted.order_date.values)):
        lst = ret_lists.get(cust)
        if not lst:
            expected[i] = 0
            continue
        n = 0
        for rd in lst:
            if np.datetime64(rd) < np.datetime64(when):
                n += 1
            else:
                break
        expected[i] = n

    actual = f_sorted.customer_prior_returns.to_numpy(int)
    n_bad = int((expected != actual).sum())
    check("independent re-derivation matches build_features.py exactly",
          n_bad == 0,
          f"{n_bad:,} of {len(actual):,} rows differ "
          f"(max delta {int(np.abs(expected - actual).max()) if n_bad else 0})")

    # The strongest single statement of the as-of rule: an order's own return must
    # never be dated at or before the order itself.
    own = f_sorted.merge(
        pairs.groupby("Invoice").InvoiceDate_r.min().rename("own_return"),
        left_on="Invoice", right_index=True, how="left")
    late = own[own.own_return.notna() & (own.own_return <= own.order_date)]
    check("no order's own return predates the order itself", len(late) == 0,
          f"{len(late)} orders have a return dated at or before purchase")

    # Rate must be exactly returns/orders where history exists.
    hist = f_sorted[f_sorted.customer_prior_orders > 0]
    recomputed = hist.customer_prior_returns / hist.customer_prior_orders
    check("prior_return_rate equals returns / orders",
          bool(np.allclose(recomputed, hist.customer_prior_return_rate, atol=1e-6)))

    # --- 9 database
    print("\n9 - database (risk.db)")
    con = sqlite3.connect(paths["risk.db"])
    cur = con.cursor()
    viol = cur.execute("PRAGMA foreign_key_check").fetchall()
    check("foreign key violations = 0", len(viol) == 0, f"got {len(viol)}")

    def count(t):
        return cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]

    n_tables = cur.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%'").fetchone()[0]
    check("all 30 schema tables created", n_tables == 30, f"got {n_tables}")
    check("risk_labels mature rows = 30,347",
          count("risk_labels WHERE is_mature=1") == EXPECTED["mature"],
          f"got {count('risk_labels WHERE is_mature=1'):,}")
    check("dataset_members = 30,347",
          count("dataset_members") == EXPECTED["mature"],
          f"got {count('dataset_members'):,}")
    db_pos = cur.execute(
        "SELECT AVG(label_risk) FROM risk_labels WHERE is_mature=1").fetchone()[0]
    check("database positive rate matches features.pkl",
          near(db_pos, pos, 1e-6), f"db {db_pos:.4%} vs features {pos:.4%}")

    db_split = dict(cur.execute(
        "SELECT split, COUNT(*) FROM dataset_members GROUP BY split").fetchall())
    check("database split counts match features.pkl",
          db_split.get("train") == len(tr) and db_split.get("test") == len(te),
          f"db {db_split} vs features train={len(tr)} test={len(te)}")

    # Chargebacks are architecturally supported but deliberately unevaluated.
    check("disputes empty by design (no dispute labels exist)",
          count("disputes") == 0, f"got {count('disputes'):,}")
    check("label_disputed is always 0",
          cur.execute("SELECT COALESCE(SUM(label_disputed),0) "
                      "FROM risk_labels").fetchone()[0] == 0)
    # risk_scores is written by score_to_db.py, which runs after Phase 2. This
    # check used to assert the table was empty, which only held while the two
    # phases were unconnected. Now: empty is fine (Phase 1 alone), populated is
    # fine too - but if it is populated it must be exactly the test split, and
    # it must join cleanly back to the Phase 1 label table.
    n_scores = count("risk_scores")
    if n_scores == 0:
        print("         risk_scores empty - Phase 2 has not been written to the db yet")
    else:
        n_test = int((feat.split == "test").sum())
        check("risk_scores covers exactly the test split",
              n_scores == n_test, f"{n_scores:,} scores vs {n_test:,} test rows")
        joined = cur.execute(
            "SELECT COUNT(*) FROM risk_scores s "
            "JOIN risk_labels l ON l.payment_id = s.payment_id").fetchone()[0]
        check("every risk_score joins to a risk_label", joined == n_scores,
              f"{joined:,} of {n_scores:,} joined")
        orphan_band = cur.execute(
            "SELECT COUNT(*) FROM risk_scores WHERE risk_band NOT IN "
            "('low','medium','high')").fetchone()[0]
        check("risk_band values are all schema-legal", orphan_band == 0)
        orphan_rec = cur.execute(
            "SELECT COUNT(*) FROM risk_scores WHERE recommendation NOT IN "
            "('allow','manual_review','hold_payout','request_verification')"
        ).fetchone()[0]
        check("recommendation values are all schema-legal", orphan_rec == 0)
        print(f"         risk_scores populated: {n_scores:,} rows, "
              f"{count('risk_score_features'):,} feature snapshots")

        # Every dependent of the tables score_to_db.py rewrites must be
        # accounted for in its delete list, or a second run dies on a foreign
        # key. It claimed to be idempotent and was not: two dependents were
        # missing, and it only ever ran after a fresh build_database.py.
        owned = {"risk_scores", "models", "threshold_config"}
        dependents = set()
        for (name, sql) in cur.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table'"):
            for parent in owned:
                if sql and f"REFERENCES {parent}(id)" in sql and name != parent:
                    dependents.add(name)
        # `reviews` is deliberately not deleted - it is a human audit trail and
        # score_to_db.py refuses to run rather than destroy it.
        cleared = {"risk_explanations", "risk_score_features", "risk_scores",
                   "model_feature_importance", "evaluations"}
        unhandled = dependents - cleared - {"reviews"}
        check("every FK dependent of the rewritten tables is handled",
              not unhandled, f"unhandled: {sorted(unhandled)}")
        n_rev = count("reviews")
        check("reviews is empty, so score_to_db.py will not refuse", n_rev == 0,
              f"{n_rev} human review rows exist - score_to_db.py will stop, by design")
    con.close()

    # --- 10 reproducibility
    print("\n10 - reproducibility")
    csv = pd.read_csv(paths["features.csv"], parse_dates=["order_date"])
    check("features.csv and features.pkl have the same shape",
          csv.shape == feat.shape, f"csv {csv.shape} vs pkl {feat.shape}")
    check("features.csv and features.pkl agree on the label",
          bool((csv.returned.to_numpy() == feat.returned.to_numpy()).all()))
    check("features.csv and features.pkl agree on every feature value",
          bool(np.allclose(csv[FEATURES].to_numpy(float),
                           feat[FEATURES].to_numpy(float), atol=1e-6)))
    check("Invoice is unique (one row per order)",
          feat.Invoice.is_unique,
          f"{int(feat.Invoice.duplicated().sum())} duplicates")
    check("feature matrix is sorted by order_date",
          bool(feat.order_date.is_monotonic_increasing))

    # --- 11 gitignore
    print("\n11 - gitignore hygiene")
    # Ask git, not the text of the file. This check used to assert that the
    # literal string "artefacts/" appeared in .gitignore - which it did, and
    # which is exactly what broke the four "!artefacts/..." negations under it:
    # git cannot re-include a file whose parent DIRECTORY is excluded, so the
    # model files the deployment needs were silently uncommittable while this
    # test reported green. A pattern being present is not a path being ignored.
    def git_ignored(path):
        """True, False, or None when git cannot answer."""
        try:
            r = subprocess.run(["git", "check-ignore", "-q", path],
                               cwd=ROOT, capture_output=True)
        except OSError:
            return None
        return r.returncode == 0 if r.returncode in (0, 1) else None

    MUST_IGNORE = [".env", "features.pkl", "orders_labeled.pkl", "risk.db",
                   "features.csv", "orders_labeled.csv",
                   "artefacts/invariance.json", "artefacts/cost_vs_threshold.png"]
    # Small, needed by the deployment, and therefore committed.
    MUST_TRACK = [".env.example", "artefacts/model.joblib",
                  "artefacts/scaler.joblib", "artefacts/threshold.json",
                  "artefacts/threshold_sweep.csv"]

    if git_ignored(".env") is None:
        print("         (git unavailable - falling back to pattern presence)")
        lines = [l.strip() for l in
                 open(os.path.join(ROOT, ".gitignore")).read().splitlines()
                 if l.strip() and not l.strip().startswith("#")]
        for pat in [".env", "*.pkl", "*.db", "features.csv",
                    "orders_labeled.csv", "*.joblib", "artefacts/*"]:
            check(f"'{pat}' is ignored", pat in lines)
        check(".env.example is NOT ignored", ".env.example" not in lines)
    else:
        for path in MUST_IGNORE:
            check(f"'{path}' is ignored", git_ignored(path) is True)
        for path in MUST_TRACK:
            check(f"'{path}' is committable", git_ignored(path) is False)
    check(".env.example exists and is committable",
          os.path.exists(os.path.join(ROOT, ".env.example")))
    env_ex = open(os.path.join(ROOT, ".env.example")).read()
    filled = re.findall(r"^([A-Z_][A-Z0-9_]*)=(.+)$", env_ex, re.M)
    # Only credential-shaped keys must be blank. Model ids, local paths and the
    # sqlite URL are configuration, not secrets, and carrying real defaults for
    # them is the point of a template - flagging those was a false positive.
    SECRET = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)", re.I)
    PLACEHOLDER = re.compile(r"^(your|<|\"|'|xxx|\.\.\.|change)", re.I)
    leaked = [k for k, v in filled
              if SECRET.search(k) and not PLACEHOLDER.match(v.strip())]
    check(".env.example contains no real-looking secret values",
          not leaked, f"suspicious keys: {leaked}")
    # A credential key present but empty is the correct state.
    creds = [k for k, _ in re.findall(r"^([A-Z_][A-Z0-9_]*)=(.*)$", env_ex, re.M)
             if SECRET.search(k)]
    check(".env.example declares at least one credential key", bool(creds),
          f"found: {creds}")

    # --- summary
    print("\n" + "=" * 72)
    print(f"  {len(_PASS)} passed, {len(_FAIL)} failed")
    if _FAIL:
        print("\n  failed checks:")
        for f in _FAIL:
            print(f"    - {f}")
        print("\n  A failure here means the pipeline changed. Investigate the")
        print("  cause before adjusting any expected number in this file.")
    else:
        print("\n  Phase 1 verified: the numbers in BUILD_PLAN.md reproduce,")
        print("  the split is chronological, and the as-of feature construction")
        print("  matches an independent re-derivation exactly.")
    print("=" * 72)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
