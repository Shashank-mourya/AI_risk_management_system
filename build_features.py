"""
Step 2: build the as-of feature matrix. Run after build_labels.py.

    python build_features.py

The rule the whole file exists to enforce: every feature for an order placed at
time T uses only events that had already happened by T. Not orders merely placed
before T - outcomes actually observed before T.

A customer's third order does not "know" their first was returned unless the
return itself occurred before the third order was placed. Counting an eventual
label as though it were known at purchase time is target leakage, and a
chronological split will not catch it.

Outputs
    features.pkl / features.csv    30,347 rows x 17 features + label + split
"""

import numpy as np
import pandas as pd

from config import (
    FEATURES, MIN_GAP_DAYS, RETURN_WINDOW_DAYS, genuine_returns,
)


def main():
    df = pd.read_pickle("retail2.pkl")
    orders = pd.read_pickle("orders_labeled.pkl")

    # --- returns
    # Rebuild the purchase->return matches so we have return dates, which is
    # what "already happened" is measured against.
    purchases = df[(~df.isC) & (df.Quantity > 0) & df.CustomerID.notna()].copy()
    returns   = df[( df.isC) & (df.Quantity < 0) & df.CustomerID.notna()].copy()
    purchases["pidx"] = purchases.index
    returns["ridx"] = returns.index

    pairs = returns[["ridx", "CustomerID", "StockCode", "InvoiceDate"]].merge(
        purchases[["pidx", "CustomerID", "StockCode", "InvoiceDate", "Invoice"]],
        on=["CustomerID", "StockCode"], suffixes=("_r", "_p"))
    pairs = pairs[pairs.InvoiceDate_p < pairs.InvoiceDate_r]
    pairs = pairs.sort_values("InvoiceDate_p").groupby("ridx", as_index=False).last()
    pairs["gap_days"] = (pairs.InvoiceDate_r - pairs.InvoiceDate_p).dt.total_seconds() / 86400
    # The same window the label uses. Counting a wider set of events as "prior
    # returns" than the label counts as "returned" would make the feature and
    # the target disagree about the word.
    pairs = genuine_returns(pairs)

    # --- order-level base
    purchases["line_value"] = purchases.Quantity * purchases.Price
    agg = purchases.groupby("Invoice").agg(
        mean_unit_price=("Price", "mean"),
        max_unit_price=("Price", "max"),
    )
    orders = orders.merge(agg, left_on="Invoice", right_index=True, how="left")

    orders["hour_of_day"] = orders.order_date.dt.hour
    orders["day_of_week"] = orders.order_date.dt.dayofweek
    orders["is_uk"] = (orders.country == "United Kingdom").astype(int)
    orders["log_order_value"] = np.log1p(orders.order_value.clip(lower=0))

    # --- customer history, true as-of
    # For each order, count the customer's prior orders and prior observed
    # returns. Prior returns are counted by the date the return happened, not
    # by the eventual label of an earlier order.
    orders = orders.sort_values("order_date").reset_index(drop=True)

    # prior order count and tenure: cheap, exact
    g = orders.groupby("customer_id")
    orders["customer_prior_orders"] = g.cumcount()
    first_seen = g.order_date.transform("min")
    orders["customer_tenure_days"] = (
        orders.order_date - first_seen).dt.total_seconds() / 86400

    # Prior observed returns, counted at order granularity.
    #
    # `pairs` is line-level: an order that sends back five stock codes produces
    # five rows. Counting those directly against customer_prior_orders (an
    # order count) mixes units, and the resulting "rate" is not a rate - it
    # reached 45.0 on this dataset. So collapse to one event per returned
    # purchase order, dated at the earliest line that came back, which is when
    # the merchant first observed that order unwinding.
    #
    # The inner merge restricts the numerator to the same labelled order
    # population the denominator counts, which is what bounds the rate to
    # [0, 1]. Only the return's occurrence date is used - never an order's
    # eventual label, so the as-of rule still holds.
    order_return_date = (pairs.groupby("Invoice").InvoiceDate_r.min()
                              .rename("return_observed_at"))
    observed = orders[["Invoice", "customer_id"]].merge(
        order_return_date, left_on="Invoice", right_index=True, how="inner")

    # per customer, binary-search the sorted list of return-observation dates
    ret_by_cust = (observed.groupby("customer_id").return_observed_at
                           .apply(lambda s: np.sort(s.values)).to_dict())
    prior_returns = np.zeros(len(orders), dtype=int)
    for i, (cust, when) in enumerate(zip(orders.customer_id.values,
                                         orders.order_date.values)):
        arr = ret_by_cust.get(cust)
        if arr is not None:
            prior_returns[i] = np.searchsorted(arr, when, side="left")
    orders["customer_prior_returns"] = prior_returns
    orders["customer_prior_return_rate"] = np.where(
        orders.customer_prior_orders > 0,
        orders.customer_prior_returns / orders.customer_prior_orders.clip(lower=1),
        -1.0,          # sentinel: no history. Cold-start path, not zero risk.
    )
    # Both sides now count orders, so this is a genuine fraction. Assert it
    # rather than trusting it: a units regression here is silent and poisons
    # every downstream metric.
    assert (orders.customer_prior_returns
            <= orders.customer_prior_orders).all(), "prior returns exceed prior orders"
    orders["is_new_customer"] = (orders.customer_prior_orders == 0).astype(int)

    # --- product-level baseline
    # Per-SKU historical return rate, built as of the split date.
    #
    # The earlier version used train-period purchases and their eventual
    # outcomes. 7.3% of those returns were observed after the split date, so a
    # test order was being scored with knowledge of returns that had not
    # happened when it was placed. The comment claimed "nothing from the test
    # window can inform it", which was true of the purchases and false of their
    # outcomes.
    #
    # Both sides are now cut at the split boundary: only purchases made before
    # it, and only returns observed before it. That is exactly what a model
    # deployed on the split date would have had. It goes mildly stale across the
    # test window, which is the honest direction - a production model has the
    # same staleness between retrains.
    split_at = orders.loc[orders.split == "train", "order_date"].max()

    known_purchases = purchases[purchases.InvoiceDate <= split_at]
    known_returns = pairs[pairs.InvoiceDate_r <= split_at]
    returned_pidx = set(known_returns.pidx)

    kp = known_purchases.copy()
    kp["ret"] = kp.index.isin(returned_pidx)
    sku = kp.groupby("StockCode").ret.agg(["sum", "size"])
    prior_mean = kp.ret.mean()
    SMOOTH = 20                       # shrink rare SKUs toward the global rate
    sku_rate = (sku["sum"] + SMOOTH * prior_mean) / (sku["size"] + SMOOTH)

    purchases["sku_rate"] = purchases.StockCode.map(sku_rate).fillna(prior_mean)
    basket = purchases.groupby("Invoice").sku_rate.agg(["mean", "max"])
    basket.columns = ["basket_sku_return_rate", "basket_max_sku_return_rate"]
    orders = orders.merge(basket, left_on="Invoice", right_index=True, how="left")

    # Price relative to the catalogue average for the same SKUs, also as of the
    # split date. Averaging over the whole dataset priced 474 SKUs entirely from
    # rows that did not exist yet, and moved 631 more by over 5%.
    sku_price = known_purchases.groupby("StockCode").Price.mean()
    global_price = known_purchases.Price.mean()
    purchases["price_ratio"] = (
        purchases.Price / purchases.StockCode.map(sku_price).fillna(global_price))
    orders = orders.merge(
        purchases.groupby("Invoice").price_ratio.mean().rename("price_vs_sku_mean"),
        left_on="Invoice", right_index=True, how="left")

    # --- emit
    out = orders[["Invoice", "customer_id", "order_date", "split", "returned"]
                 + FEATURES].copy()
    out["returned"] = out.returned.astype(int)
    out = out.dropna(subset=FEATURES)

    out.to_pickle("features.pkl")
    out.to_csv("features.csv", index=False)

    tr, te = out[out.split == "train"], out[out.split == "test"]
    print(f"""
{'='*64}
  FEATURE MATRIX
{'='*64}
  rows                 {len(out):>8,}      features {len(FEATURES)}
  train                {len(tr):>8,}      {tr.returned.mean():.2%} positive ({tr.returned.sum():,})
  test                 {len(te):>8,}      {te.returned.mean():.2%} positive ({te.returned.sum():,})
  new customers        {out.is_new_customer.mean():>8.1%}      (cold-start, rate sentinel = -1)
{'='*64}""")
    print("\ncorrelation with label (train only):")
    print(tr[FEATURES + ["returned"]].corr()["returned"]
            .drop("returned").sort_values(key=abs, ascending=False).round(3).to_string())
    print("\nwrote features.pkl  <- train from this")
    print("wrote features.csv  <- inspect this")


if __name__ == "__main__":
    main()
