"""
AI Risk Manager
The cost model. Single source of truth, imported by everything that costs a
decision: the training notebook, evaluate_model.py, and the Streamlit app.

    from cost_model import total_cost_pence, P_STAR, describe

WHY THIS IS ITS OWN MODULE
--------------------------
It used to live in two places that disagreed. The notebook charged
`fp*c_fp + fn*c_fn`; evaluate_model.py charged review time on every flag and
only credited a caught return with partial prevention. Two cost models means
two different "optimal" thresholds, and the graded bar here is honest metrics
including false-positive cost - so the number a reviewer is shown must come
from one place.

THE FORMULA
-----------
Flagging an order is not free and catching a return is not the same as
preventing it. Both were missing from the original model.

    total = (tp + fp) * REVIEW            every flagged order costs analyst time
          +       fp  * FRICTION          a wrongly-flagged good order annoys a customer
          +       fn  * COST_RETURN       a missed return, absorbed in full
          +       tp  * COST_RETURN * (1 - PREVENTION)
                                          a caught return is only sometimes stopped

The last term is the one people forget. If intervention only works 30% of the
time, a true positive still costs 70% of a miss, and a model cannot be credited
with savings it did not produce.

WHAT IS AND IS NOT DEFENDED
---------------------------
GROUNDED IN THIS DATASET (measured, see build_features.py output):
  - the two representative order values, below

INDUSTRY ASSUMPTIONS, NOT MEASURED (this dataset carries no cost data at all -
no shipping, no margin, no refund records). They are named variables precisely
so a reviewer can argue with the assumption instead of a magic number, and
`threshold_sensitivity.png` shows how far the operating point moves when they
are wrong:
  - GOODS_RECOVERY_RATE, PSP_FEE_RATE, RETURN_LOGISTICS_PENCE
  - COST_REVIEW_PENCE, ABANDON_RATE, CONTRIBUTION_MARGIN_RATE
  - PREVENTION_RATE

Do not present the pound figures as measured outcomes. Present the *shape* -
that the optimum sits well below 0.5, and why - which is what survives being
wrong about any single constant.
"""

import numpy as np
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------------
# Representative order values - MEASURED on training rows only.
#
# Two different values, because they price two different populations. The cost
# of a missed return depends on what returned orders are worth; the cost of a
# false alarm depends on what good orders are worth. On this data those differ
# by a third, and collapsing them to one median would misprice both sides.
# ---------------------------------------------------------------------------
RETURNED_ORDER_VALUE_PENCE = 38815   # median order_value, returned train orders (GBP 388.15)
KEPT_ORDER_VALUE_PENCE     = 28559   # median order_value, kept train orders    (GBP 285.59)

# ---------------------------------------------------------------------------
# Assumptions. Every one of these is arguable; none is measured here.
# ---------------------------------------------------------------------------
GOODS_RECOVERY_RATE      = 0.65   # share of value recovered by reselling returned goods
PSP_FEE_RATE             = 0.02   # payment fee on the original sale, not returned on a refund
RETURN_LOGISTICS_PENCE   = 850    # collection + inbound handling + restocking, per return

COST_REVIEW_PENCE        = 500    # analyst time to review ONE flagged order
ABANDON_RATE             = 0.08   # share of wrongly-flagged customers who walk
CONTRIBUTION_MARGIN_RATE = 0.22   # gross margin lost when they do

PREVENTION_RATE          = 0.30   # share of CAUGHT returns actually prevented

# ---------------------------------------------------------------------------
# Derived costs
# ---------------------------------------------------------------------------
COST_RETURN_PENCE = int(round(
    RETURNED_ORDER_VALUE_PENCE * (1 - GOODS_RECOVERY_RATE)   # value not recovered
    + RETURNED_ORDER_VALUE_PENCE * PSP_FEE_RATE              # sunk payment fee
    + RETURN_LOGISTICS_PENCE                                 # moving the goods back
))

COST_FRICTION_PENCE = int(round(
    KEPT_ORDER_VALUE_PENCE * ABANDON_RATE * CONTRIBUTION_MARGIN_RATE
))

# Analytic break-even. Flag when the expected cost of flagging is lower:
#   E[flag]    = REVIEW + (1-p)*FRICTION + p*COST_RETURN*(1-PREVENTION)
#   E[no flag] = p*COST_RETURN
#   =>  flag iff  p > (REVIEW + FRICTION) / (FRICTION + COST_RETURN*PREVENTION)
#
# This is the threshold the cost model implies with no data at all. The
# empirical sweep should land near it; a large gap means the probabilities are
# miscalibrated, and that is worth knowing.
P_STAR = ((COST_REVIEW_PENCE + COST_FRICTION_PENCE)
          / (COST_FRICTION_PENCE + COST_RETURN_PENCE * PREVENTION_RATE))


def total_cost_pence(y_true, y_pred, c_review=None, c_friction=None,
                     c_return=None, prevention=None):
    """
    Total cost in pence of one set of decisions.

    The optional overrides exist for the sensitivity analysis, which needs to
    re-cost the same predictions under different assumptions without mutating
    module state.
    """
    c_review    = COST_REVIEW_PENCE    if c_review    is None else c_review
    c_friction  = COST_FRICTION_PENCE  if c_friction  is None else c_friction
    c_return    = COST_RETURN_PENCE    if c_return    is None else c_return
    prevention  = PREVENTION_RATE      if prevention  is None else prevention

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float((tp + fp) * c_review
                 + fp * c_friction
                 + fn * c_return
                 + tp * c_return * (1 - prevention))


def total_cost_pence_per_order(y_true, y_pred, order_value_pence):
    """
    The same model, but priced on each order's OWN value instead of a
    representative median.

    Returned orders here are worth more than kept ones and the tail is long
    (max GBP 77k), so a median-priced total understates what misses actually
    cost. This is reported alongside the headline as a cross-check, not used to
    pick the threshold - a per-order optimum would be tuned to the test set's
    particular large orders, which is exactly the kind of quiet overfitting
    this repo is supposed to avoid.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    v = np.asarray(order_value_pence, dtype=float)

    cost_return = (v * (1 - GOODS_RECOVERY_RATE) + v * PSP_FEE_RATE
                   + RETURN_LOGISTICS_PENCE)
    friction = v * ABANDON_RATE * CONTRIBUTION_MARGIN_RATE

    flagged = y_pred == 1
    return float(
        flagged.sum() * COST_REVIEW_PENCE
        + friction[flagged & (y_true == 0)].sum()
        + cost_return[~flagged & (y_true == 1)].sum()
        + cost_return[flagged & (y_true == 1)].sum() * (1 - PREVENTION_RATE)
    )


def as_dict():
    """Every input and derived figure, for threshold.json and the write-up."""
    return {
        "measured": {
            "returned_order_value_pence": RETURNED_ORDER_VALUE_PENCE,
            "kept_order_value_pence": KEPT_ORDER_VALUE_PENCE,
        },
        "assumed": {
            "goods_recovery_rate": GOODS_RECOVERY_RATE,
            "psp_fee_rate": PSP_FEE_RATE,
            "return_logistics_pence": RETURN_LOGISTICS_PENCE,
            "cost_review_pence": COST_REVIEW_PENCE,
            "abandon_rate": ABANDON_RATE,
            "contribution_margin_rate": CONTRIBUTION_MARGIN_RATE,
            "prevention_rate": PREVENTION_RATE,
        },
        "derived": {
            "cost_return_pence": COST_RETURN_PENCE,
            "cost_friction_pence": COST_FRICTION_PENCE,
            "analytic_break_even": round(P_STAR, 6),
        },
        "formula": ("(tp+fp)*review + fp*friction + fn*cost_return "
                    "+ tp*cost_return*(1-prevention)"),
    }


def describe():
    """Human-readable summary. Printed by the notebook and the evaluator."""
    return f"""cost model (pence)
  MEASURED on train rows
    median value, returned orders   {RETURNED_ORDER_VALUE_PENCE:>8,}   (GBP {RETURNED_ORDER_VALUE_PENCE/100:,.2f})
    median value, kept orders       {KEPT_ORDER_VALUE_PENCE:>8,}   (GBP {KEPT_ORDER_VALUE_PENCE/100:,.2f})
  ASSUMED (industry figures, not in this dataset)
    goods recovery rate             {GOODS_RECOVERY_RATE:>8.0%}
    payment fee, not refunded       {PSP_FEE_RATE:>8.0%}
    reverse logistics               {RETURN_LOGISTICS_PENCE:>8,}
    manual review, per flag         {COST_REVIEW_PENCE:>8,}
    abandon rate on a false alarm   {ABANDON_RATE:>8.0%}
    contribution margin             {CONTRIBUTION_MARGIN_RATE:>8.0%}
    prevention rate on a catch      {PREVENTION_RATE:>8.0%}
  DERIVED
    cost of a return                {COST_RETURN_PENCE:>8,}   (GBP {COST_RETURN_PENCE/100:,.2f})
    cost of a false alarm           {COST_FRICTION_PENCE:>8,}   (GBP {COST_FRICTION_PENCE/100:,.2f})
    review charged on EVERY flag    {COST_REVIEW_PENCE:>8,}
    analytic break-even p*          {P_STAR:>8.4f}
  a caught return still costs {(1-PREVENTION_RATE):.0%} of a missed one"""


if __name__ == "__main__":
    print(describe())
