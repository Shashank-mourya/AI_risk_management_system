"""
Synthetic stand-in for features.pkl, emitted in the real schema.

    python make_synthetic_dataset.py

features.pkl is the real training input and the notebook uses it by default.
This generator is the fallback for when the real matrix is not on hand: a Kaggle
kernel where uploading a 139 MB risk.db or rerunning the downloader is not worth
it, CI and smoke tests that must not depend on a network download, and
exercising test_model.py without a full pipeline run.

It emits the same 17 feature columns, the same id column (`Invoice`), the same
single -1 sentinel and the same train/test split semantics as build_features.py,
so the notebook needs no branching beyond one flag.

It is not evaluation data. Any precision/recall/AUC computed on these rows
measures the generator, not the model, and must not be reported; artefacts
trained on it are stamped REPORTABLE=false.

Leakage discipline mirrors the real pipeline. Return outcomes are drawn from
latent customer and SKU propensities plus the order's own attributes, never from
the computed history features. The as-of features are then built in a second,
strictly chronological pass in which a prior return counts only if its return
date precedes the current order date - never by an earlier order's eventual
label. Both the customer-history and the basket-SKU features obey this.

Gap constants come from the measured distribution of the real data, so the
synthetic gaps reproduce it (p50 10d / p90 84d / p95 ~150d, the 90-day window
capturing ~90.6% of returns).
"""

import numpy as np
import pandas as pd

SEED = 42
N_CUSTOMERS = 5_900          # real: 5,942 with a usable id
N_SKUS = 4_600               # real: 4,631 products loaded

WINDOW_START = pd.Timestamp("2009-12-01")
WINDOW_END = pd.Timestamp("2011-12-09")

from config import MIN_GAP_DAYS, RETURN_WINDOW_DAYS, TRAIN_FRACTION  # noqa: E402

# Solved from the measured gap distribution: median = exp(mu) = 10d, p90 = exp(mu + 1.2816*sigma) = 84d,
# which also gives p95 ~= 153d and P(gap <= 90) ~= 0.907.
GAP_MU = np.log(10.0)
GAP_SIGMA = (np.log(84.0) - GAP_MU) / 1.2816

# P(1 <= gap <= 90) under that lognormal is ~0.824, so an eventual-return rate of
# ~0.2255 lands the observed positive rate near the real 18.58%.
TARGET_EVENTUAL_RETURN = 0.2255

# Smoothing for as-of SKU return rates. The real build_features.py bottoms out at
# 0.000465 rather than a sentinel, i.e. it falls back to a global prior for SKUs
# with no history, so this mirrors that rather than inventing a second sentinel.
SKU_SMOOTHING = 20.0

# The exact 17 columns build_features.py emits, in its order.
FEATURES = [
    "order_value", "log_order_value", "n_lines", "total_quantity",
    "mean_unit_price", "max_unit_price", "price_vs_sku_mean",
    "hour_of_day", "day_of_week", "is_uk",
    "customer_prior_orders", "customer_prior_returns",
    "customer_prior_return_rate", "customer_tenure_days", "is_new_customer",
    "basket_sku_return_rate", "basket_max_sku_return_rate",
]
assert len(FEATURES) == 17

# Only this one carries -1. "No history" is a different state from "never
# returned" and must not be imputed to 0.
SENTINEL_FEATURES = ["customer_prior_return_rate"]
SENTINEL = -1.0


def build_populations(rng):
    customers = pd.DataFrame({
        "customer_id": np.arange(N_CUSTOMERS),
        "propensity": rng.beta(2.0, 6.5, N_CUSTOMERS),
        "is_uk": (rng.random(N_CUSTOMERS) < 0.90).astype(int),
        "basket_scale": rng.lognormal(5.7, 0.8, N_CUSTOMERS),
        "n_orders": np.maximum(1, rng.lognormal(1.55, 0.95, N_CUSTOMERS).round()).astype(int),
    })
    # per-SKU return propensity and a catalogue price
    sku_propensity = rng.beta(1.8, 7.0, N_SKUS)
    sku_price = rng.lognormal(1.0, 0.9, N_SKUS)
    return customers, sku_propensity, sku_price


def build_orders(rng, customers, sku_propensity, sku_price):
    span = (WINDOW_END - WINDOW_START).days
    cid, offs = [], []
    for c in customers.itertuples():
        first = rng.uniform(0, span * 0.85)
        gaps = rng.exponential(42.0, max(0, c.n_orders - 1))
        o = np.concatenate([[first], first + np.cumsum(gaps)])
        o = o[o <= span]
        cid.extend([c.customer_id] * len(o))
        offs.extend(o.tolist())

    o = pd.DataFrame({"customer_id": np.array(cid), "day_offset": np.array(offs)})
    o = o.merge(customers[["customer_id", "propensity", "is_uk", "basket_scale"]],
                on="customer_id", how="left")
    o["order_date"] = (WINDOW_START + pd.to_timedelta(o.day_offset, unit="D")).dt.floor("min")
    o = o.sort_values("order_date").reset_index(drop=True)
    n = len(o)

    o["n_lines"] = np.maximum(1, rng.poisson(20.0, n))
    o["total_quantity"] = np.maximum(
        1, (o.n_lines * rng.lognormal(2.0, 0.7, n)).round()).astype(float)

    # ---- basket composition: which SKUs are in each order ------------------
    baskets = [rng.choice(N_SKUS, size=min(int(k), N_SKUS), replace=False)
               for k in o.n_lines]
    o["basket"] = baskets
    o["basket_mean_prop"] = [float(sku_propensity[b].mean()) for b in baskets]
    o["basket_max_prop"] = [float(sku_propensity[b].max()) for b in baskets]

    cat_price = np.array([float(sku_price[b].mean()) for b in baskets])
    o["order_value"] = np.round(o.basket_scale * rng.lognormal(0.0, 0.5, n), 2)
    o["log_order_value"] = np.round(np.log1p(o.order_value), 6)
    o["mean_unit_price"] = np.round(o.order_value / o.total_quantity.clip(lower=1), 2)
    o["max_unit_price"] = np.round(o.mean_unit_price * rng.uniform(1.1, 6.0, n), 2)
    o["price_vs_sku_mean"] = np.round(o.mean_unit_price / np.maximum(cat_price, 0.01), 4)

    o["hour_of_day"] = np.clip(rng.normal(12.5, 2.6, n).round(), 6, 20).astype(int)
    o["day_of_week"] = o.order_date.dt.dayofweek.astype(int)
    o["is_uk"] = o.is_uk.astype(int)
    return o


def draw_outcomes(rng, o):
    """
    Eventual-return outcome from LATENT propensities + order attributes only.

    No computed history feature is read here, so the as-of features built in the
    next pass cannot be a restatement of the label.

    Weights are set so the basket-SKU signal dominates, matching the real
    correlation ranking where basket_max_sku_return_rate (0.258) and
    log_order_value (0.200) lead.
    """
    n = len(o)

    def z(v):
        return (v - np.mean(v)) / np.std(v)

    logit = (
        3.0 * (o.propensity.to_numpy() - 0.235)
        + 0.95 * z(o.basket_max_prop.to_numpy())
        + 0.45 * z(o.basket_mean_prop.to_numpy())
        + 0.60 * z(o.log_order_value.to_numpy())
        + 0.30 * z(o.n_lines.to_numpy().astype(float))
        - 0.18 * z(np.log(o.mean_unit_price.to_numpy() + 0.01))
        - 0.20 * o.is_uk.to_numpy()
        + rng.normal(0, 0.35, n)
    )

    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(logit + mid)))).mean() > TARGET_EVENTUAL_RETURN:
            hi = mid
        else:
            lo = mid
    q = 1 / (1 + np.exp(-(logit + (lo + hi) / 2)))

    eventual = rng.random(n) < q
    gap = rng.lognormal(GAP_MU, GAP_SIGMA, n)

    o = o.copy()
    o["return_gap_days"] = np.where(eventual, gap, np.nan)
    o.loc[eventual & (gap < MIN_GAP_DAYS), "return_gap_days"] = np.nan  # clerical

    has = o.return_gap_days.notna()
    o["return_date"] = pd.NaT
    o.loc[has, "return_date"] = (o.loc[has, "order_date"]
                                 + pd.to_timedelta(o.loc[has, "return_gap_days"], unit="D"))
    o["returned"] = (has & (o.return_gap_days <= RETURN_WINDOW_DAYS)).astype(int)
    return o


def build_asof_features(o):
    """
    Strictly chronological second pass.

    Customer history AND per-SKU history count an earlier return only when that
    return's date precedes the current order date.
    """
    o = o.sort_values("order_date").reset_index(drop=True)
    n = len(o)

    prior_orders = np.zeros(n, dtype=int)
    prior_returns = np.zeros(n, dtype=int)
    prior_rate = np.full(n, SENTINEL)
    tenure = np.zeros(n)
    basket_rate = np.zeros(n)
    basket_max = np.zeros(n)

    first_seen, c_dates, c_rets = {}, {}, {}
    sku_orders = np.zeros(N_SKUS)          # prior orders containing this SKU
    sku_returns = np.zeros(N_SKUS)         # prior returned orders containing it
    pending = []                           # (return_date, basket) not yet counted
    pending_i = 0

    global_returns = 0.0
    global_orders = 0.0

    for i, r in enumerate(o.itertuples()):
        now = r.order_date

        # release every return that had actually happened before this order
        while pending_i < len(pending) and pending[pending_i][0] < now:
            _, b = pending[pending_i]
            sku_returns[b] += 1
            global_returns += 1
            pending_i += 1

        # ---- customer history ------------------------------------------------
        cid = r.customer_id
        dates = c_dates.get(cid, [])
        rets = c_rets.get(cid, [])
        k = len(dates)
        prior_orders[i] = k
        if k > 0:
            nret = int(np.searchsorted(np.array(rets), now, side="left")) if rets else 0
            prior_returns[i] = nret
            prior_rate[i] = nret / k
            tenure[i] = (now - first_seen[cid]).total_seconds() / 86400.0
        else:
            first_seen[cid] = now

        # ---- basket SKU history (smoothed toward the global prior) ----------
        b = r.basket
        prior = (global_returns / global_orders) if global_orders > 0 else 0.0
        rates = (sku_returns[b] + SKU_SMOOTHING * prior) / (sku_orders[b] + SKU_SMOOTHING)
        basket_rate[i] = float(rates.mean())
        basket_max[i] = float(rates.max())

        # ---- advance state ---------------------------------------------------
        dates.append(now)
        c_dates[cid] = dates
        if pd.notna(r.return_date):
            rets.append(r.return_date)
            rets.sort()
            c_rets[cid] = rets
            pending.append((r.return_date, b))
            pending.sort(key=lambda t: t[0])
        sku_orders[b] += 1
        global_orders += 1

    o["customer_prior_orders"] = prior_orders
    o["customer_prior_returns"] = prior_returns
    o["customer_prior_return_rate"] = np.round(prior_rate, 6)
    o["customer_tenure_days"] = np.round(tenure, 4)
    o["is_new_customer"] = (prior_orders == 0).astype(int)
    o["basket_sku_return_rate"] = np.round(basket_rate, 6)
    o["basket_max_sku_return_rate"] = np.round(basket_max, 6)
    return o


def main():
    rng = np.random.default_rng(SEED)
    customers, sku_prop, sku_price = build_populations(rng)
    o = build_orders(rng, customers, sku_prop, sku_price)
    o = draw_outcomes(rng, o)
    o = build_asof_features(o)

    # maturity, then an 80/20 chronological split - matching build_labels.py,
    # which emits only mature orders.
    end = o.order_date.max()
    cutoff = end - pd.Timedelta(days=RETURN_WINDOW_DAYS)
    n_all = len(o)
    mature = o[o.order_date <= cutoff].sort_values("order_date").reset_index(drop=True)
    split_date = mature.order_date.iloc[int(len(mature) * TRAIN_FRACTION)].normalize()
    mature["split"] = np.where(mature.order_date < split_date, "train", "test")

    mature["Invoice"] = "SYN" + mature.index.astype(str).str.zfill(6)
    out = mature[["Invoice", "customer_id", "order_date", "split", "returned"] + FEATURES].copy()
    out["customer_id"] = out.customer_id.astype(float)

    out.to_csv("data/synthetic_features.csv", index=False)
    out.to_pickle("data/synthetic_features.pkl")

    tr, te = out[out.split == "train"], out[out.split == "test"]
    print("\n" + "=" * 64)
    print("  SYNTHETIC FEATURE MATRIX  -  NOT evaluation data")
    print("=" * 64)
    print(f"  rows                 {len(out):>8,}      features {len(FEATURES)}")
    print(f"  train                {len(tr):>8,}      {tr.returned.mean():.2%} positive ({tr.returned.sum():,})")
    print(f"  test                 {len(te):>8,}      {te.returned.mean():.2%} positive ({te.returned.sum():,})")
    print(f"  immature, excluded   {n_all - len(mature):>8,}")
    print(f"  new customers        {out.is_new_customer.mean():>8.1%}      (cold-start, rate sentinel = -1)")
    print(f"  positive rate        {out.returned.mean():>8.2%}")
    print(f"  split date           {str(split_date.date()):>8}")
    print(f"  order value (GBP)    median {out.order_value.median():.2f}   mean {out.order_value.mean():.2f}")
    g = o.return_gap_days.dropna()
    print(f"  gap p50/p90/p95      {g.quantile(.5):.1f}d / {g.quantile(.9):.1f}d / {g.quantile(.95):.1f}d")
    print(f"  90-day window        {(g <= 90).mean():.1%} of returns")
    print("=" * 64)
    print("\ncorrelation with label (train only):")
    print(tr[FEATURES + ["returned"]].corr()["returned"]
          .drop("returned").sort_values(key=abs, ascending=False).round(3).to_string())
    print("\nwrote data/synthetic_features.csv")
    print("wrote data/synthetic_features.pkl")


if __name__ == "__main__":
    main()
