"""
AI Risk Manager
Shared constants. Import these; never redefine them.

WHY THIS FILE EXISTS
--------------------
This is the third time the same failure has shown up in this repo. The cost
model lived in two places that disagreed, so two different "optimal" thresholds
were simultaneously true. The scoring path lived in two places, and the copy
calling itself "the single scoring path" sat in a test file nothing imports -
emitting recommendations the database schema would have rejected.

`MIN_GAP_DAYS` and `RETURN_WINDOW_DAYS` were defined independently in SIX files,
one of them carrying the comment "must match build_labels.py" - which is a
comment doing a constant's job. A label horizon that drifts between the labeller,
the feature builder and the database is not a style problem: it silently changes
what the word "returned" means depending on which script you ask.

THE LABEL DEFINITION
--------------------
An order is RETURNED if at least one of its lines is reversed by a credit note
raised between MIN_GAP_DAYS and RETURN_WINDOW_DAYS after the purchase.

Both bounds are load-bearing and both were chosen from the data:

  lower bound   11.1% of matched credit notes land on the same calendar day as
                the purchase. Those are clerical corrections, not customer
                returns; without the floor the model learns to predict the
                retailer's own data-entry errors.

  upper bound   THIS ONE WAS MISSING and it mattered. Maturity guarantees every
                labelled order had 90 days to be observed, but the label used to
                count returns at ANY horizon. So an order from Dec 2009 was
                watched for 667 days and one from Sep 2011 for 90, and the
                positive rate rose with the length of the watch: 17.6% in the
                shortest-window quintile against 20.3% in the longest. That is
                not risk, it is exposure time - and because the split is
                chronological, train sat in the long-window end and test in the
                short one, manufacturing a train/test gap out of nothing.
                Capping the label at the same 90 days maturity already
                guarantees makes the two horizons agree.
"""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------- the .env
# Loaded HERE and nowhere else. Every module that needs a credential or an
# env-overridable path imports it from config, for the same reason the label
# window and the cost model live in one file: a setting read in three places
# is three settings that can disagree. Importing config is what makes .env
# take effect - so the scorer, the explainer and the app all import it.
#
# Real environment variables WIN over .env (override=False): a deployment that
# sets GROQ_API_KEY in the process environment must not be overridden by a
# stale file left in the working tree.
ENV_PATH = os.path.join(ROOT, ".env")

try:
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH, override=False)
except ImportError:      # python-dotenv absent: real env vars still work
    pass


def env(name, default=None):
    """Read a setting, treating an empty value in .env as unset."""
    v = os.environ.get(name)
    return v if v not in (None, "") else default


# --------------------------------------------------------------------- LLM
# Explanation layer only. A missing key removes prose, never a decision -
# explain.py raises ExplanationError and writes status='failed'.
GROQ_API_KEY = env("GROQ_API_KEY")
GROQ_MODEL = env("GROQ_MODEL", "openai/gpt-oss-120b")
# The two comparison models for `python explain.py --invariance`: three
# different models, three different paragraphs, one identical decision.
GROQ_MODEL_SMALL = env("GROQ_MODEL_SMALL", "openai/gpt-oss-20b")
GROQ_MODEL_ALT = env("GROQ_MODEL_ALT",
                     "qwen/qwen3.8-27b")

# --------------------------------------------------------------- the dataset
SOURCE_URL = (
    "https://raw.githubusercontent.com/allanvc/onlineretail2/master/"
    "data/onlineretail2.rda"
)

# Pinned so the pipeline cannot silently retrain on a different file. This is a
# plain HTTPS fetch of a third-party artefact on a branch that can be force
# -pushed; without a checksum, "the numbers moved" and "upstream changed" are
# indistinguishable. Verified after download, before anything parses it.
SOURCE_SHA256 = "be2480b1fcb1fa123f90d731f63ab3cf44f6c97cba0a38eabee93759d411c6f7"
SOURCE_FILE = "onlineretail2.rda"

# ---------------------------------------------------------------- the label
MIN_GAP_DAYS = 1.0        # below this it is a clerical fix, not a return
RETURN_WINDOW_DAYS = 90   # the label horizon AND the maturity horizon

# Chronological, never random: a random split leaks the future into training
# through the customer-history features.
TRAIN_FRACTION = 0.80

# --------------------------------------------------------------- the database
MERCHANT_ID = "acc_demo01"
DATASET_ID = "ds_v1"
CURRENCY = "GBP"          # UK retailer: stored amounts are in PENCE, not paise.
                          # This is the provenance of the data and does not
                          # change. Rupees are a DISPLAY layer only - see
                          # cost_model.GBP_TO_INR / fmt_inr().

def _abs(path):
    """Relative paths in .env are relative to the repo, not to the CWD."""
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def _sqlite_path(url, default):
    """
    DATABASE_URL -> a file path. SQLite only, on purpose: nothing in this repo
    speaks Postgres, and silently accepting a postgres:// URL would look like
    support that does not exist.
    """
    if not url:
        return default
    if not url.startswith("sqlite:"):
        raise ValueError(
            f"DATABASE_URL={url!r}: only sqlite:/// URLs are supported. "
            "The demo runs off a local risk.db.")
    return _abs(url.split("sqlite:///", 1)[-1] or default)


# ------------------------------------------------------------------- paths
RETAIL_PKL = os.path.join(ROOT, "retail2.pkl")
ORDERS_PKL = os.path.join(ROOT, "orders_labeled.pkl")
FEATURES_PKL = os.path.join(ROOT, "features.pkl")
DB_PATH = _sqlite_path(env("DATABASE_URL"), os.path.join(ROOT, "risk.db"))
SCHEMA_SQL = os.path.join(ROOT, "AI_Risk_Manager_schema_v3.sql")
ART_DIR = os.path.join(ROOT, "artefacts")

# Artefact paths are env-overridable so a demo can point at a different trained
# model without editing code. Defaults are what train_model.ipynb writes.
MODEL_PATH = _abs(env("MODEL_PATH", os.path.join(ART_DIR, "model.joblib")))
THRESHOLD_PATH = _abs(env("THRESHOLD_PATH", os.path.join(ART_DIR, "threshold.json")))
SCALER_PATH = _abs(env("SCALER_PATH", os.path.join(ART_DIR, "scaler.joblib")))

# The 17 as-of features emitted by build_features.py, in order. predict.py
# reads the authoritative list from threshold.json; this copy is for the
# builders and tests, and test_phase1.py asserts the two agree.
FEATURES = [
    "order_value", "log_order_value", "n_lines", "total_quantity",
    "mean_unit_price", "max_unit_price", "price_vs_sku_mean",
    "hour_of_day", "day_of_week", "is_uk",
    "customer_prior_orders", "customer_prior_returns",
    "customer_prior_return_rate", "customer_tenure_days", "is_new_customer",
    "basket_sku_return_rate", "basket_max_sku_return_rate",
]

# Only this one uses -1. "No history" and "never returned" are different states
# and must not be collapsed.
SENTINEL_FEATURES = ["customer_prior_return_rate"]
SENTINEL_VALUE = -1


def genuine_returns(matches):
    """
    Apply the label definition to a table of purchase->return matches.

    One function, so the labeller, the feature builder, the database loader and
    the test suite cannot disagree about what counts as a return.

    `matches` needs a `gap_days` column; returns the filtered frame.
    """
    return matches[(matches.gap_days >= MIN_GAP_DAYS)
                   & (matches.gap_days <= RETURN_WINDOW_DAYS)]
