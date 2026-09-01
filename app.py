"""
AI Risk Manager
Phase 4: the demo surface.

    streamlit run app.py

One file. No React, no component library, no build step.

THE POINT OF THIS APP
---------------------
Not "here is a model". The threshold tab is the argument: a reviewer moves the
cost assumptions and watches the optimal operating point move with them. That
makes the case that the threshold is a business decision the cost model settles,
not a constant the model emits - which is the graded bar on this track.

WHAT IT DOES NOT DO
-------------------
It does not take actions. A score produces a RECOMMENDATION; acting on one needs
a named reviewer (hard rule #6). And nothing the explanation layer returns can
change a score (hard rule #4) - the app renders the decision from predict.py and
the prose from explain.py separately, and never reconciles them.
"""

import json
import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import cost_model as cm
from predict import RECOMMENDATION, Scorer, band_bounds, risk_band

# Imported for the .env load as much as for the path: the explanation panel
# below needs GROQ_API_KEY to have reached the process environment.
from config import DB_PATH

ROOT = os.path.dirname(os.path.abspath(__file__))
DB = DB_PATH
FEATURES_PKL = os.path.join(ROOT, "features.pkl")

st.set_page_config(page_title="AI Risk Manager - Return Risk",
                   page_icon="📦", layout="wide")

# ---------------------------------------------------------------- palette
# Validated categorical/diverging steps (see the data-viz reference palette).
# Dark steps are selected for the dark surface, not an automatic flip.
LIGHT = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#dcdbd6",
             series="#2a78d6", up="#e34948", down="#2a78d6", neutral="#f0efec")
DARK = dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#3a3a37",
            series="#3987e5", up="#e66767", down="#3987e5", neutral="#383835")

# Status colors are reserved and always ship with a label, never color alone.
STATUS = {"allow": ("#0ca30c", "✅"), "manual_review": ("#fab219", "⚠️"),
          "hold_payout": ("#d03b3b", "⛔")}


def theme():
    try:
        return DARK if st.get_option("theme.base") == "dark" else LIGHT
    except Exception:
        return LIGHT


def style_axes(ax, t):
    ax.set_facecolor("none")
    ax.figure.patch.set_alpha(0)
    for s in ax.spines.values():
        s.set_color(t["grid"])
    ax.tick_params(colors=t["ink2"], labelsize=8)
    ax.xaxis.label.set_color(t["ink2"])
    ax.yaxis.label.set_color(t["ink2"])
    ax.title.set_color(t["ink"])
    ax.grid(alpha=.25, color=t["grid"], lw=.8)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ loading
@st.cache_resource
def load_scorer():
    return Scorer()


@st.cache_data
def load_features():
    return pd.read_pickle(FEATURES_PKL)


@st.cache_data
def test_predictions():
    """Held-out probabilities. Cached so the threshold slider is instant."""
    s = load_scorer()
    df = load_features()
    te = df[df.split == "test"].sort_values("order_date").reset_index(drop=True)
    p = s.score_batch(te[s.features].to_numpy(float))
    return te, p, te.returned.to_numpy(int)


scorer = load_scorer()
meta = scorer.meta

if not scorer.reportable:
    st.error("These artefacts were trained on SYNTHETIC data. "
             "No number on this page is reportable.")

st.title("AI Risk Manager — return-risk scorer")
st.caption(
    f"UCI Online Retail II · {meta['winner']} · operating threshold "
    f"{scorer.threshold} · amounts in GBP")

tab_score, tab_cost, tab_metrics = st.tabs(
    ["Score an order", "Cost & threshold", "Held-out evidence"])


# ========================================================== 1 SCORE AN ORDER
with tab_score:
    te, p_test, y_test = test_predictions()
    t = theme()

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.subheader("Order")
        st.caption(
            "Start from a real held-out order, then change anything. Typing 17 "
            "features from scratch is not a demo.")

        band_pick = st.selectbox(
            "Load an example", ["highest risk", "typical", "lowest risk",
                                "a return the model caught",
                                "a return the model missed"])
        if band_pick == "highest risk":
            idx = int(np.argmax(p_test))
        elif band_pick == "lowest risk":
            idx = int(np.argmin(p_test))
        elif band_pick == "typical":
            idx = int(np.argsort(p_test)[len(p_test) // 2])
        elif band_pick == "a return the model caught":
            hit = np.where((y_test == 1) & (p_test >= scorer.threshold))[0]
            idx = int(hit[len(hit) // 2]) if len(hit) else 0
        else:
            miss = np.where((y_test == 1) & (p_test < scorer.threshold))[0]
            idx = int(miss[len(miss) // 2]) if len(miss) else 0

        row = te.iloc[idx]
        st.caption(f"Invoice {row.Invoice} · {row.order_date:%Y-%m-%d} · "
                   f"actual outcome: **{'returned' if row.returned else 'kept'}** "
                   "(shown for context; the model never sees it)")

        order = {}
        c1, c2 = st.columns(2)
        with c1:
            order["order_value"] = st.number_input(
                "Order value (GBP)", value=float(row.order_value), step=10.0)
            order["n_lines"] = st.number_input(
                "Line items", value=int(row.n_lines), step=1, min_value=1)
            order["total_quantity"] = st.number_input(
                "Total quantity", value=int(row.total_quantity), step=1)
            order["mean_unit_price"] = st.number_input(
                "Mean unit price", value=float(row.mean_unit_price), step=0.5)
            order["max_unit_price"] = st.number_input(
                "Max unit price", value=float(row.max_unit_price), step=0.5)
            order["is_uk"] = int(st.checkbox("United Kingdom", value=bool(row.is_uk)))
        with c2:
            order["customer_prior_orders"] = st.number_input(
                "Customer's prior orders", value=int(row.customer_prior_orders),
                step=1, min_value=0,
                help="Orders this customer placed BEFORE this one. Only orders "
                     "already placed by this moment count — nothing later is "
                     "visible to the model.")
            order["customer_prior_returns"] = st.number_input(
                "Prior orders that were returned",
                value=int(row.customer_prior_returns),
                step=1, min_value=0,
                help="How many of those prior orders had at least one item sent "
                     "back. Counted as ORDERS, not line items, so it can never "
                     "exceed the number above — and only returns already "
                     "observed by this moment count.")
            order["customer_tenure_days"] = st.number_input(
                "Customer tenure (days)", value=float(row.customer_tenure_days),
                step=10.0)
            order["basket_sku_return_rate"] = st.slider(
                "Basket mean SKU return rate", 0.0, 1.0,
                float(row.basket_sku_return_rate), 0.005)
            order["basket_max_sku_return_rate"] = st.slider(
                "Basket worst SKU return rate", 0.0, 1.0,
                float(row.basket_max_sku_return_rate), 0.005)
            order["hour_of_day"] = st.slider("Hour of day", 0, 23,
                                             int(row.hour_of_day))

        order["day_of_week"] = int(row.day_of_week)
        order["price_vs_sku_mean"] = float(row.price_vs_sku_mean)
        order["log_order_value"] = float(np.log1p(max(order["order_value"], 0)))

        # The sentinel is a state, not a number. If the customer has no history
        # the rate must stay -1; imputing 0 would say "never returned", which is
        # a different and much safer-looking claim.
        prior_orders = order["customer_prior_orders"]
        order["is_new_customer"] = int(prior_orders == 0)
        order["customer_prior_return_rate"] = (
            -1.0 if prior_orders == 0
            else min(order["customer_prior_returns"], prior_orders) / prior_orders)
        if prior_orders == 0:
            st.caption("No prior orders → `customer_prior_return_rate = -1` "
                       "(cold start, **not** 'never returned')")

    with right:
        out = scorer.score_order(order, top_n=8)
        st.subheader("Decision")

        m1, m2, m3 = st.columns(3)
        m1.metric("Return probability", f"{out['risk_probability']:.1%}")
        m2.metric("Risk band", out["risk_band"].upper())
        m3.metric("Threshold", f"{out['threshold_applied']:.2f}")

        # Native callouts rather than hand-rolled HTML. The old version passed
        # unsafe_allow_html=True; everything it interpolated came from closed
        # dicts, so it was safe as written - but it left a raw-HTML sink one
        # careless edit away from rendering a model-authored or order-derived
        # string. These carry the same colour semantics with no sink at all,
        # and they follow the viewer's theme for free.
        callout = {"allow": st.success, "manual_review": st.warning,
                   "hold_payout": st.error}[out["recommendation"]]
        icon = STATUS[out["recommendation"]][1]
        callout(
            f"{icon} **{out['recommendation'].replace('_', ' ').title()}** — "
            f"a recommendation, not an action. Acting on it requires a named "
            f"reviewer.")

        st.caption(
            f"Bands sit at the threshold ({band_bounds(scorer.threshold)[0]:.2f}) "
            f"and at twice it ({band_bounds(scorer.threshold)[1]:.2f}), so they "
            f"move when the operating point does. "
            f"Base rate is {y_test.mean():.1%}.")

        # ---- contributions: diverging, because sign is the whole message ----
        st.subheader("What drove this score")
        contrib = out["top_features"]
        if contrib:
            d = pd.DataFrame(contrib).iloc[::-1]
            fig, ax = plt.subplots(figsize=(6.4, 3.6))
            colors = [t["up"] if v > 0 else t["down"] for v in d.contribution]
            ax.barh(d.feature, d.contribution, color=colors, height=.62)
            ax.axvline(0, color=t["ink2"], lw=1)
            style_axes(ax, t)
            ax.set_xlabel("contribution to log-odds  ← lowers risk · raises risk →")
            # barh already set one tick per feature; restyling them via
            # set_yticklabels alone detaches labels from a fixed locator.
            ax.tick_params(axis="y", labelsize=8)
            for y_i, (v, val) in enumerate(zip(d.contribution, d.value)):
                ax.annotate(f"{v:+.2f}", (v, y_i), fontsize=8, color=t["ink2"],
                            va="center", ha="left" if v > 0 else "right",
                            xytext=(4 if v > 0 else -4, 0),
                            textcoords="offset points")
            ax.margins(x=.20)
            st.pyplot(fig, width='stretch')
            plt.close(fig)
            st.caption(
                "For logistic regression this is exact — coefficient × standardised "
                "value, the additive terms of the log-odds. Not a post-hoc "
                "approximation. It is the main reason the linear model was worth keeping.")
        else:
            st.info("The shipped model has no linear decomposition, so no exact "
                    "per-feature contribution is shown rather than an invented one.")

        # ---- the LLM paragraph, strictly downstream of the decision ---------
        st.subheader("Explanation")
        st.caption("Written by an LLM from the finished score above. It cannot "
                   "change any number on this page.")
        if st.button("Generate explanation"):
            con = None
            try:
                import explain as ex
                if not os.path.exists(DB):
                    st.warning("risk.db not found — run build_database.py and "
                               "score_to_db.py.")
                else:
                    con = ex.connect(DB)
                    # Explanations are cached against SCORED orders in the
                    # database. A hand-edited order has no score row, so the
                    # stored score for this invoice is used.
                    r = con.execute(
                        "SELECT id FROM risk_scores WHERE order_id = ?",
                        (f"order_{row.Invoice}",)).fetchone()
                    if r is None:
                        st.warning("This order has no stored score. Run "
                                   "`python score_to_db.py`.")
                    else:
                        with st.spinner("calling the model…"):
                            res = ex.explain(con, r["id"])
                        if res["status"] == "ready":
                            st.success(res["explanation"])

                            if abs(res["risk_probability"]
                                   - out["risk_probability"]) > 1e-9:
                                st.caption(
                                    "Note: the form above has been edited, so this "
                                    "paragraph describes the STORED score "
                                    f"({res['risk_probability']:.1%}), not the "
                                    "edited one. The stored score is what the "
                                    "explanation was written against.")
                        else:
                            st.warning("No explanation available.")
                            # ex.explain already redacts anything credential
                            # -shaped out of the error before it gets here.
                            st.caption(f"Reason: {res.get('error')}")
                            st.caption(
                                "The score above is unchanged. A failure in this "
                                "layer removes prose, never a decision.")
            except Exception as e:  # noqa: BLE001
                # Redact here too: this path catches errors raised BEFORE
                # ex.explain got the chance to scrub them.
                try:
                    msg = ex.redact(f"{type(e).__name__}: {e}")
                except Exception:  # noqa: BLE001
                    msg = type(e).__name__
                st.warning(f"No explanation available — {msg}")
                st.caption("The score above is unchanged.")
            finally:
                if con is not None:
                    con.close()


# ======================================================= 2 COST & THRESHOLD
with tab_cost:
    te, p_test, y_test = test_predictions()
    t = theme()

    st.subheader("The threshold is a business decision")
    st.markdown(
        "Move the cost assumptions and watch the optimum move. That is the whole "
        "argument: **0.5 is not a threshold, it is a default** — and the right "
        "operating point falls out of what a mistake costs, not out of the model.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Measured** (training rows)")
        v_ret = st.number_input("Median value, returned orders (GBP)",
                                value=cm.RETURNED_ORDER_VALUE_PENCE / 100, step=10.0)
        v_kept = st.number_input("Median value, kept orders (GBP)",
                                 value=cm.KEPT_ORDER_VALUE_PENCE / 100, step=10.0)
    with c2:
        st.markdown("**Assumed** — cost of a return")
        recovery = st.slider("Goods recovery rate", 0.0, 1.0,
                             cm.GOODS_RECOVERY_RATE, 0.01)
        logistics = st.number_input("Reverse logistics (GBP)",
                                    value=cm.RETURN_LOGISTICS_PENCE / 100, step=0.5)
        prevention = st.slider("Share of caught returns actually prevented",
                               0.01, 1.0, cm.PREVENTION_RATE, 0.01)
    with c3:
        st.markdown("**Assumed** — cost of a false alarm")
        review = st.number_input("Manual review, per flag (GBP)",
                                 value=cm.COST_REVIEW_PENCE / 100, step=0.5)
        abandon = st.slider("Abandon rate on a false alarm", 0.0, 0.5,
                            cm.ABANDON_RATE, 0.01)
        margin = st.slider("Contribution margin", 0.0, 1.0,
                           cm.CONTRIBUTION_MARGIN_RATE, 0.01)

    c_return = (v_ret * 100 * (1 - recovery) + v_ret * 100 * cm.PSP_FEE_RATE
                + logistics * 100)
    c_friction = v_kept * 100 * abandon * margin
    c_review = review * 100
    p_star = (c_review + c_friction) / (c_friction + c_return * prevention)

    grid = np.round(np.arange(0.01, 0.995, 0.01), 4)
    costs = np.array([
        cm.total_cost_pence(y_test, (p_test >= thr).astype(int),
                            c_review=c_review, c_friction=c_friction,
                            c_return=c_return, prevention=prevention)
        for thr in grid])
    best_i = int(np.argmin(costs))
    best_thr, best_cost = float(grid[best_i]), float(costs[best_i])

    cost_none = cm.total_cost_pence(
        y_test, np.zeros(len(y_test), int), c_review=c_review,
        c_friction=c_friction, c_return=c_return, prevention=prevention)
    cost_ship = cm.total_cost_pence(
        y_test, (p_test >= scorer.threshold).astype(int), c_review=c_review,
        c_friction=c_friction, c_return=c_return, prevention=prevention)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cost-optimal threshold", f"{best_thr:.2f}",
              delta=f"{best_thr - scorer.threshold:+.2f} vs shipped",
              delta_color="off")
    k2.metric("Analytic break-even", f"{p_star:.3f}")
    k3.metric("Cost at that point", f"£{best_cost/100:,.0f}")
    k4.metric("Saved vs flagging nothing",
              f"{(cost_none - best_cost)/cost_none:.1%}")

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(grid, costs / 100, color=t["series"], lw=2)
    ax.plot(best_thr, best_cost / 100, "o", color=t["series"], ms=9, zorder=5)
    ax.annotate(f"optimum {best_thr:.2f}\n£{best_cost/100:,.0f}",
                (best_thr, best_cost / 100), textcoords="offset points",
                xytext=(12, 16), fontsize=9, color=t["series"], ha="left",
                bbox=dict(boxstyle="round,pad=.25", fc=t["surface"],
                          ec=t["series"], lw=.8, alpha=.9))
    ax.axhline(cost_none / 100, color=t["ink2"], ls="--", lw=1.1)
    ax.annotate("flag nothing (absorb every return)", (0.99, cost_none / 100),
                ha="right", va="bottom", fontsize=8, color=t["ink2"])
    ax.axvline(scorer.threshold, color=t["ink2"], lw=1, alpha=.55)
    ax.annotate(f"shipped {scorer.threshold:.2f}", (scorer.threshold, ax.get_ylim()[1]),
                rotation=90, va="top", ha="right", fontsize=8, color=t["ink2"])
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("total cost on 6,070 held-out orders (GBP)")
    ax.margins(y=.14)
    style_axes(ax, t)
    st.pyplot(fig, width='stretch')
    plt.close(fig)

    st.caption(
        f"Formula: `{cm.as_dict()['formula']}`. Reviewing a flagged order costs "
        f"money whether or not the flag was right, and a caught return is only "
        f"prevented {prevention:.0%} of the time — both terms are missing from the "
        f"naive `fp·c_fp + fn·c_fn`, and without them the optimum collapses to "
        f"'flag almost everything'.")

    if abs(best_thr - scorer.threshold) > 0.02:
        st.info(
            f"Under **these** assumptions the optimum is {best_thr:.2f}, but the "
            f"shipped model still operates at {scorer.threshold:.2f} "
            f"(£{cost_ship/100:,.0f}). Moving a slider does not silently "
            f"re-deploy anything — changing the operating point is a decision "
            f"someone signs off on.")

    with st.expander("Which assumptions actually matter?"):
        rows = []
        for name, kw in [("cost of a return ×0.5", dict(c_return=c_return * .5)),
                         ("cost of a return ×2", dict(c_return=c_return * 2)),
                         ("prevention 10%", dict(prevention=.10)),
                         ("prevention 90%", dict(prevention=.90)),
                         ("review cost ×5", dict(c_review=c_review * 5))]:
            base = dict(c_review=c_review, c_friction=c_friction,
                        c_return=c_return, prevention=prevention)
            base.update(kw)
            cs = [cm.total_cost_pence(y_test, (p_test >= th).astype(int), **base)
                  for th in grid]
            th_i = int(np.argmin(cs))
            yhat = (p_test >= grid[th_i]).astype(int)
            rows.append({"assumption": name,
                         "optimal threshold": float(grid[th_i]),
                         "flag rate": float(yhat.mean()),
                         "recall": float(((yhat == 1) & (y_test == 1)).sum()
                                         / max(y_test.sum(), 1))})
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True,
                     width='stretch')
        st.caption("Only two inputs really move the answer: what a return costs, "
                   "and how well intervention works.")


# ======================================================= 3 HELD-OUT EVIDENCE
with tab_metrics:
    te, p_test, y_test = test_predictions()
    h = meta["holdout"]
    base = y_test.mean()
    t = theme()

    st.subheader("Held-out performance")
    st.caption(
        f"{h['n_test']:,} orders after {h['split_date']}, never seen in training. "
        f"The split is chronological — every training order precedes every test order.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{h['precision']:.3f}", f"{h['precision']/base:.2f}× base rate",
              delta_color="off")
    c2.metric("Recall", f"{h['recall']:.3f}")
    c3.metric("ROC-AUC", f"{h['roc_auc']:.3f}")
    c4.metric("PR-AUC", f"{h['pr_auc']:.3f}", f"{h['pr_auc']/base:.2f}× base rate",
              delta_color="off")

    st.warning(
        f"**Accuracy is {h['accuracy']:.3f} — below the {1-base:.3f} you get by "
        f"flagging nothing.** That is expected at a recall-heavy operating point, "
        f"and it is why accuracy is not the selection metric here. The base rate "
        f"is {base:.2%}; quote it beside every number above or none of them.")

    cm_c1, cm_c2 = st.columns([1, 1.3], gap="large")
    with cm_c1:
        st.markdown("**Confusion matrix** at threshold "
                    f"{meta['chosen_threshold']}")
        conf = h["confusion"]
        st.dataframe(pd.DataFrame(
            [[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]],
            index=["actually kept", "actually returned"],
            columns=["predicted keep", "predicted return"]),
            width='stretch')
    with cm_c2:
        st.markdown("**Cost model inputs**")
        cmd = cm.as_dict()
        st.dataframe(pd.DataFrame(
            [{"input": k, "value": v, "status": "measured"}
             for k, v in cmd["measured"].items()]
            + [{"input": k, "value": v, "status": "assumed"}
               for k, v in cmd["assumed"].items()]),
            hide_index=True, width='stretch')

    st.markdown("---")
    st.markdown("#### What this does not measure")
    st.markdown(f"""
- **Chargebacks are unevaluated.** No public dataset carries disputes, so
  `label_disputed` is always 0. The dispute path exists in the schema and the API
  and reports **no metrics**. Synthetic dispute labels would measure our own
  generator, so declining to report them *is* the honesty bar being met.
- **The seven assumed cost inputs are not measured.** This dataset contains no
  cost data at all. Read the *shape* — the optimum sits near {meta['analytic_break_even']:.2f},
  well below the 0.5 default, and moves slowly — not the pound figures.
- **The source is a UK wholesale gift retailer.** Median order £304, customers
  mostly businesses, so return behaviour is B2B-flavoured. Amounts stay in GBP:
  converting to INR would imply the data is Indian when it is not.
- **22.8% of source rows have no customer ID** and are excluded entirely, because
  a return that cannot be attributed to a customer cannot become a label.
- No payment method, addresses or discount data exist in the source, so those
  planned features do not exist.
""")

    with st.expander("Model selection — why the simpler model shipped"):
        st.write(meta["rationale"])
        st.caption(
            f"Selection rule: {meta['selection_rule']}. The "
            f"{meta['marginal_gain_threshold']:.0%} bar was committed to before "
            f"the result was known; measured gain was "
            f"{meta['relative_gain_over_logreg']:+.2%}.")
