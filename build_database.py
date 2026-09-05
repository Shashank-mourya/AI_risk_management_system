"""
Step 3: create risk.db and load Online Retail II into it. Run after
build_labels.py.

    python build_database.py

Builds the schema from AI_Risk_Manager_schema_v3.sql and loads the nine tables
that carry data. disputes and payment_context stay empty on purpose (schema doc,
section 10.2); the rest are written at runtime by the model, evaluation and
review layers.

The database file is regenerable - don't commit it.
"""

import os
import re
import sqlite3
import numpy as np
import pandas as pd

from config import (
    CURRENCY, DATASET_ID, MERCHANT_ID, MIN_GAP_DAYS, RETURN_WINDOW_DAYS,
    genuine_returns,
)

DB = "risk.db"
SCHEMA = "AI_Risk_Manager_schema_v3.sql"

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def q(ident):
    """
    Quote a SQL identifier that cannot be bound as a parameter.

    Table names here come from our own schema file, not from user input, so
    this is defence in depth rather than a live hole - but an f-string building
    SQL should never be the thing standing between a schema change and an
    injection, so it is validated and quoted.
    """
    if not _IDENT.match(ident):
        raise ValueError(f"refusing to interpolate {ident!r} into SQL")
    return f'"{ident}"'


def epoch(s):
    """
    pandas datetime -> integer Unix epoch SECONDS, per Razorpay convention.

    Do not use `.astype('int64') // 10**9` here. This dataset arrives as
    datetime64[s] (second resolution), so that cast already yields seconds and
    the division collapses every timestamp to 1. Casting to datetime64[s]
    explicitly first makes the unit unambiguous whatever the source resolution.
    """
    return pd.to_datetime(s).astype("datetime64[s]").astype("int64")


def pence(v):
    return np.round(np.asarray(v, dtype=float) * 100).astype("int64")


def main():
    if os.path.exists(DB):
        os.remove(DB)

    df = pd.read_pickle("retail2.pkl")
    orders_lbl = pd.read_pickle("orders_labeled.pkl")   # mature orders + split

    con = sqlite3.connect(DB)
    con.executescript(open(SCHEMA).read().split("--  REFERENCE QUERY 1")[0])
    con.execute("PRAGMA foreign_keys=ON")

    end = df.InvoiceDate.max()
    now = int(pd.Timestamp.now('UTC').timestamp())

    # --- merchants
    con.execute(
        "INSERT INTO merchants VALUES (?,?,?,?,?,?)",
        (MERCHANT_ID, "Online Retail II (UCI) demo merchant",
         "demo@riskmgr.dev", CURRENCY, RETURN_WINDOW_DAYS,
         int(df.InvoiceDate.min().timestamp())))

    purchases = df[(~df.isC) & (df.Quantity > 0) & df.CustomerID.notna()].copy()
    purchases["line_value"] = purchases.Quantity * purchases.Price

    # --- customers
    cust = purchases.groupby("CustomerID").InvoiceDate.min().reset_index()
    cust["id"] = "cust_" + cust.CustomerID.astype(int).astype(str)
    con.executemany(
        "INSERT INTO customers VALUES (?,?,?,?,?,?,?,?)",
        [(r.id, MERCHANT_ID, None, None, None,
          int(r.InvoiceDate.timestamp()), int(r.InvoiceDate.timestamp()), now)
         for r in cust.itertuples()])

    # --- orders
    o = purchases.groupby("Invoice").agg(
        cust=("CustomerID", "first"), date=("InvoiceDate", "min"),
        country=("Country", "first"), value=("line_value", "sum")).reset_index()
    o["oid"] = "order_" + o.Invoice.astype(str)
    o["cid"] = "cust_" + o.cust.astype(int).astype(str)
    o["ts"] = epoch(o.date)
    o["amt"] = pence(o.value)
    con.executemany(
        "INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(r.oid, MERCHANT_ID, int(r.amt), int(r.amt), 0, CURRENCY,
          str(r.Invoice), "paid", 1, None, r.country, None, int(r.ts))
         for r in o.itertuples()])

    # --- products
    prod = purchases.groupby("StockCode").agg(
        name=("Description", "first"), price=("Price", "mean")).reset_index()
    prod["pid"] = "prod_" + prod.StockCode.astype(str)
    con.executemany(
        "INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?)",
        [(r.pid, MERCHANT_ID, str(r.StockCode), r.name, "uncategorised",
          None, int(round(r.price * 100)), "none", now) for r in prod.itertuples()])

    # --- order_items
    purchases["oitem"] = "oitem_" + purchases.index.astype(str)
    con.executemany(
        "INSERT INTO order_items VALUES (?,?,?,?,?,?,?,?,?)",
        [(r.oitem, "order_" + str(r.Invoice), "prod_" + str(r.StockCode),
          str(r.StockCode), "uncategorised", int(round(r.Price * 100)),
          int(r.Quantity), None, 0)
         for r in purchases.itertuples()])

    # --- payments
    # The source has no payment layer: one captured payment per order.
    # `method` is a constant and is excluded from the feature set.
    con.executemany(
        "INSERT INTO payments (id,merchant_id,order_id,customer_id,amount,currency,"
        "status,method,captured,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        [("pay_" + str(r.Invoice), MERCHANT_ID, r.oid, r.cid, int(r.amt),
          CURRENCY, "captured", "card", 1, int(r.ts)) for r in o.itertuples()])

    # --- refunds
    returns = df[(df.isC) & (df.Quantity < 0) & df.CustomerID.notna()].copy()
    purchases["pidx"] = purchases.index
    returns["ridx"] = returns.index
    pairs = returns[["ridx", "CustomerID", "StockCode", "InvoiceDate", "Quantity"]].merge(
        purchases[["pidx", "CustomerID", "StockCode", "InvoiceDate", "Invoice", "Price"]],
        on=["CustomerID", "StockCode"], suffixes=("_r", "_p"))
    pairs = pairs[pairs.InvoiceDate_p < pairs.InvoiceDate_r]
    pairs = pairs.sort_values("InvoiceDate_p").groupby("ridx", as_index=False).last()
    pairs["gap"] = (pairs.InvoiceDate_r - pairs.InvoiceDate_p).dt.total_seconds() / 86400
    # Same label window as build_labels.py, via the same function, so risk_labels
    # cannot disagree with orders_labeled.pkl about what "returned" means.
    pairs = pairs.rename(columns={"gap": "gap_days"})
    genuine = genuine_returns(pairs).copy()
    genuine["rts"] = epoch(genuine.InvoiceDate_r)
    con.executemany(
        "INSERT INTO refunds (id,merchant_id,payment_id,amount,currency,status,"
        "refund_reason,created_at) VALUES (?,?,?,?,?,?,?,?)",
        [(f"rfnd_{r.ridx}", MERCHANT_ID, "pay_" + str(r.Invoice),
          int(round(abs(r.Quantity) * r.Price * 100)), CURRENCY, "processed",
          "customer_return", int(r.rts)) for r in genuine.itertuples()])

    # --- risk_labels
    # label_disputed is always 0: there is no dispute data. label_matured_at
    # follows the merchant's return window; immature orders are excluded, not
    # counted as clean.
    first_ret = genuine.groupby("Invoice").rts.min()
    cutoff_ts = int((end - pd.Timedelta(days=RETURN_WINDOW_DAYS)).timestamp())
    rows = []
    for r in o.itertuples():
        ret_ts = first_ret.get(r.Invoice)
        refunded = int(pd.notna(ret_ts))
        matured = int(r.ts + RETURN_WINDOW_DAYS * 86400 <= end.timestamp())
        rows.append(("pay_" + str(r.Invoice), MERCHANT_ID, refunded, 0, refunded,
                     int(ret_ts) if refunded else None,
                     int(r.ts) + RETURN_WINDOW_DAYS * 86400, matured,
                     None if matured else "immature", 1, now))
    con.executemany("INSERT INTO risk_labels VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)

    # --- datasets & split freeze
    import hashlib
    members = [(DATASET_ID, "pay_" + str(r.Invoice), r.split)
               for r in orders_lbl.itertuples()]
    h = hashlib.sha256("".join(sorted(m[1] + m[2] for m in members)).encode()).hexdigest()
    tr = sum(1 for m in members if m[2] == "train")
    con.execute("INSERT INTO datasets VALUES (?,?,?,?,?,?,?,?,?,?)",
                (DATASET_ID, MERCHANT_ID, "online-retail-ii chronological 80/20",
                 int(orders_lbl[orders_lbl.split == "train"].order_date.max().timestamp()),
                 1, tr, len(members) - tr,
                 float(orders_lbl.returned.mean()), h, now))
    con.executemany("INSERT INTO dataset_members VALUES (?,?,?)", members)

    con.commit()

    # --- verify
    print(f"\n{'='*58}\n  {DB}  —  {os.path.getsize(DB)/1e6:.1f} MB\n{'='*58}")
    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    for t in tables:
        n = con.execute(f"SELECT COUNT(*) FROM {q(t)}").fetchone()[0]
        tag = ""
        if t in ("disputes", "payment_context"):
            tag = "  <- empty by design"
        elif n == 0:
            tag = "  <- written at runtime"
        print(f"  {t:<28} {n:>9,}{tag}")

    bad = con.execute("PRAGMA foreign_key_check").fetchall()
    print(f"\n  foreign key violations: {len(bad)}")
    pr = con.execute("""SELECT ROUND(AVG(label_risk)*100,2) FROM risk_labels
                        WHERE is_mature=1""").fetchone()[0]
    print(f"  positive rate (mature): {pr}%")
    print(f"  split hash:             {h[:16]}…\n")
    con.close()


if __name__ == "__main__":
    main()
