"""
Step 1: fetch Online Retail II, build the return label, emit a labelled and
chronologically-split order table.

    python build_labels.py

Every cutoff here is derived from the data rather than assumed - the values in
CONFIG came out of profiling this dataset, see the printout at the end.

Outputs
    retail2.pkl          raw dataset
    orders_labeled.pkl   one row per order, with `returned` and `split`
"""

import hashlib
import os

import numpy as np
import pandas as pd

from config import (
    MIN_GAP_DAYS, RETURN_WINDOW_DAYS, SOURCE_FILE, SOURCE_SHA256, SOURCE_URL,
    TRAIN_FRACTION, genuine_returns,
)

# Every cutoff lives in config.py so the labeller, the feature builder, the
# database loader and the test suite cannot disagree about what "returned"
# means. They used to define these independently in six files, one of them
# under the comment "must match build_labels.py".


def _verify(path):
    """
    Fail loudly if the file is not the one this repo was measured on.

    Without this, "the numbers moved" and "upstream force-pushed" look
    identical, and every verified count in BUILD_PLAN.md quietly stops
    meaning anything.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != SOURCE_SHA256:
        raise SystemExit(f"""
CHECKSUM MISMATCH for {path}
  expected {SOURCE_SHA256}
  got      {got}
The upstream file changed, or the download was corrupted. Delete it to
re-fetch. Do NOT update the constant to match without re-verifying every
number in BUILD_PLAN.md.""")
    return got


def load() -> pd.DataFrame:
    """Fetch the dataset once and normalise column names and dtypes."""
    import pyreadr

    path = SOURCE_FILE
    if not os.path.exists(path):
        import urllib.request
        print(f"downloading {SOURCE_URL}")
        # Verify before the file becomes the one the pipeline reads, so a
        # corrupted fetch cannot be mistaken for a cached good one next run.
        tmp = path + ".part"
        urllib.request.urlretrieve(SOURCE_URL, tmp)
        _verify(tmp)
        os.replace(tmp, path)
    print(f"source sha256 {_verify(path)[:16]}... verified")

    df = list(pyreadr.read_r(path).values())[0]
    df.columns = ["Invoice", "StockCode", "Description", "Quantity",
                  "InvoiceDate", "Price", "CustomerID", "Country"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    # A credit note is an invoice number prefixed 'C'. This is the dataset's
    # own marker for goods sent back.
    df["isC"] = df.Invoice.astype(str).str.upper().str.startswith("C")
    return df


def match_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Match each credit-note line back to the purchase it reverses.

    Returns ALL matches with their gap; the label window is applied by
    config.genuine_returns, so every consumer applies the same one.

    A credit note is matched to the NEAREST PRIOR purchase of the same stock
    code by the same customer. Purchases strictly before the credit note only —
    the same discipline that keeps the feature pipeline leakage-free.

    Note what is deliberately NOT matched: negative-quantity rows that are not
    credit notes. Those carry descriptions like 'damages', 'check', 'smashed'
    and are merchant-side inventory write-offs, not customer returns. In the
    v3 schema they map to refund_reason='merchant_error', which is excluded
    from the label.
    """
    purchases = df[(~df.isC) & (df.Quantity > 0) & df.CustomerID.notna()].copy()
    returns   = df[( df.isC) & (df.Quantity < 0) & df.CustomerID.notna()].copy()
    purchases["pidx"] = purchases.index
    returns["ridx"] = returns.index

    pairs = returns[["ridx", "CustomerID", "StockCode", "InvoiceDate"]].merge(
        purchases[["pidx", "CustomerID", "StockCode", "InvoiceDate", "Invoice"]],
        on=["CustomerID", "StockCode"], suffixes=("_r", "_p"))

    pairs = pairs[pairs.InvoiceDate_p < pairs.InvoiceDate_r]
    pairs = pairs.sort_values("InvoiceDate_p").groupby("ridx", as_index=False).last()
    pairs["gap_days"] = (
        pairs.InvoiceDate_r - pairs.InvoiceDate_p
    ).dt.total_seconds() / 86400.0
    return pairs


def build_orders(df: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one row per order and attach the label."""
    # The label horizon and the maturity horizon must be the same window.
    # Counting returns at any horizon while guaranteeing only 90 days of
    # observation makes the positive rate a function of how long each order
    # happened to be watched - see config.py.
    genuine = genuine_returns(matches)
    returned_lines = set(genuine.pidx)

    p = df[(~df.isC) & (df.Quantity > 0) & df.CustomerID.notna()].copy()
    p["line_returned"] = p.index.isin(returned_lines)
    p["line_value"] = p.Quantity * p.Price

    orders = p.groupby("Invoice").agg(
        customer_id=("CustomerID", "first"),
        order_date=("InvoiceDate", "min"),
        country=("Country", "first"),
        n_lines=("StockCode", "size"),
        total_quantity=("Quantity", "sum"),
        order_value=("line_value", "sum"),
        returned=("line_returned", "max"),
    ).reset_index()

    # Maturity: an order too close to the end of the data has not had time to
    # be returned. Excluded, not labelled negative.
    end = df.InvoiceDate.max()
    cutoff = end - pd.Timedelta(days=RETURN_WINDOW_DAYS)
    orders["is_mature"] = orders.order_date <= cutoff

    mature = orders[orders.is_mature].copy()

    # Chronological split. Never random: a random split leaks future
    # information into training through the customer-history features.
    split_at = mature.order_date.quantile(TRAIN_FRACTION)
    mature["split"] = np.where(mature.order_date <= split_at, "train", "test")
    return orders, mature, end, cutoff, split_at


def main():
    df = load()
    df.to_pickle("retail2.pkl")

    non_credit_negatives = ((df.Quantity < 0) & ~df.isC).sum()
    matches = match_returns(df)
    orders, mature, end, cutoff, split_at = build_orders(df, matches)

    tr = mature[mature.split == "train"]
    te = mature[mature.split == "test"]

    print(f"""
{'='*66}
  DATASET
{'='*66}
  rows                        {len(df):>10,}
  date range                  {df.InvoiceDate.min():%Y-%m-%d} -> {end:%Y-%m-%d}
  unique customers            {df.CustomerID.nunique():>10,}
  unique stock codes          {df.StockCode.nunique():>10,}
  rows with no CustomerID     {df.CustomerID.isna().sum():>10,}  ({df.CustomerID.isna().mean():.1%})
                              excluded: returns cannot be attributed

{'='*66}
  LABEL CONSTRUCTION
{'='*66}
  credit-note lines           {(df.isC & (df.Quantity<0) & df.CustomerID.notna()).sum():>10,}
  matched to a purchase       {len(matches):>10,}  ({len(matches)/(df.isC & (df.Quantity<0) & df.CustomerID.notna()).sum():.1%})
  dropped, same-day clerical  {(matches.gap_days<MIN_GAP_DAYS).sum():>10,}
  dropped, beyond the window  {(matches.gap_days>RETURN_WINDOW_DAYS).sum():>10,}  outside the {RETURN_WINDOW_DAYS}d label horizon
  non-credit negative rows    {non_credit_negatives:>10,}  merchant write-offs,
                              never counted as customer returns

  purchase -> return gap      p50 {matches.gap_days.median():.1f}d   p90 {matches.gap_days.quantile(.9):.1f}d   p95 {matches.gap_days.quantile(.95):.1f}d
  {RETURN_WINDOW_DAYS}-day window captures      {(matches.gap_days<=RETURN_WINDOW_DAYS).mean():.1%} of matched returns
  LABEL                       returned within [{MIN_GAP_DAYS:.0f}, {RETURN_WINDOW_DAYS}] days
                              the same window maturity guarantees

{'='*66}
  ORDERS
{'='*66}
  total (customer known)      {len(orders):>10,}
  maturity cutoff             {cutoff:%Y-%m-%d}
  immature, excluded          {(~orders.is_mature).sum():>10,}
  mature, labelled            {len(mature):>10,}

  POSITIVE RATE               {mature.returned.mean():>10.2%}   ({mature.returned.sum():,} returned orders)

{'='*66}
  CHRONOLOGICAL SPLIT  (at {split_at:%Y-%m-%d})
{'='*66}
  train                       {len(tr):>10,} orders   {tr.returned.mean():.2%} positive  ({tr.returned.sum():,})
  test                        {len(te):>10,} orders   {te.returned.mean():.2%} positive  ({te.returned.sum():,})

  order value (GBP)           median {mature.order_value.median():.2f}   mean {mature.order_value.mean():.2f}
  customers with >1 order     {(mature.groupby('customer_id').size()>1).mean():.1%}
{'='*66}
""")
    # Pickle is the file to train from: it round-trips dtypes exactly, so
    # order_date stays a datetime and `returned` stays a bool.
    mature.to_pickle("orders_labeled.pkl")

    # CSV is the file to look at — open it in Excel, diff it, paste from it.
    # Reading it back needs explicit parsing, so do not train from this one:
    #     pd.read_csv("orders_labeled.csv", parse_dates=["order_date"])
    mature.to_csv("orders_labeled.csv", index=False)

    print(f"wrote orders_labeled.pkl   ({len(mature):,} rows)  <- train from this")
    print(f"wrote orders_labeled.csv   ({len(mature):,} rows)  <- inspect this")


if __name__ == "__main__":
    main()
