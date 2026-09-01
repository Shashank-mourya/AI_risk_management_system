-- ============================================================================
--  AI Risk Manager — Return & Chargeback Risk Scorer
--  Database Schema v3 (FINAL)
--  Razorpay Hackathon, Track 02
--
--  v3 finalises v2 against the chosen training dataset (UCI Online Retail II).
--  Changes from v2: orders.country added; dataset-binding notes recorded in the
--  design document (which tables load, which stay empty, and why).
--
--  Target: SQLite 3.31+ (local development) and PostgreSQL 12+ (Supabase).
--
--  Portability notes:
--    * JSON columns are declared TEXT. On PostgreSQL, change TEXT -> JSONB
--      for every column commented "-- json".
--    * BOOLEAN is stored as INTEGER 0/1 in SQLite; PostgreSQL uses it natively.
--    * All timestamps are INTEGER Unix epoch seconds, matching Razorpay's
--      public API, which returns created_at and respond_by as integers.
--    * Amounts are INTEGER in the smallest currency unit (paise for INR).
--
--  Convention: tables marked [RZP] mirror a Razorpay public API entity.
--              Tables marked [EXT] are our own additions.
-- ============================================================================

-- Tables are declared in dependency order: the file runs top-to-bottom on
-- both SQLite and PostgreSQL with no forward references.


-- ==========================================================================
--  LAYER A — TENANCY
--  Every row in every other table belongs to exactly one merchant.
-- ==========================================================================

-- [EXT] Merchant accounts. Every row in every other table belongs to exactly
-- one merchant; the API never accepts merchant_id as a parameter, it is
-- always derived from the API key.
CREATE TABLE merchants (
    id               TEXT PRIMARY KEY,          -- acc_xxxxxxxxxxxx
    name             TEXT    NOT NULL,
    email            TEXT    NOT NULL,
    currency         TEXT    NOT NULL DEFAULT 'INR',
    dispute_window_days INTEGER NOT NULL DEFAULT 120,  -- drives label maturity
    created_at       INTEGER NOT NULL
);

-- [EXT] API credentials. role gates the admin-only endpoints.
CREATE TABLE api_keys (
    id               TEXT PRIMARY KEY,          -- key_xxxxxxxxxxxx
    merchant_id      TEXT    NOT NULL REFERENCES merchants(id),
    key_hash         TEXT    NOT NULL UNIQUE,   -- argon2/bcrypt, never the raw key
    role             TEXT    NOT NULL CHECK (role IN ('standard','admin')),
    active           BOOLEAN NOT NULL DEFAULT 1,
    created_at       INTEGER NOT NULL,
    last_used_at     INTEGER
);


-- ==========================================================================
--  LAYER B — CUSTOMER IDENTITY
--  Keyed on a synthetic id, not on a phone number.
-- ==========================================================================

-- [EXT] Customer identity. Keyed on a synthetic id, not on a phone number:
-- the API exposes cust_xxxx identifiers, contact details change, and PII does
-- not belong in a primary key or a URL.
CREATE TABLE customers (
    id               TEXT PRIMARY KEY,          -- cust_xxxxxxxxxxxx
    merchant_id      TEXT    NOT NULL REFERENCES merchants(id),
    email            TEXT,
    contact          TEXT,
    account_created_at INTEGER,                 -- merchant's signup timestamp
    first_seen_at    INTEGER NOT NULL,          -- first transaction observed
    last_seen_at     INTEGER,
    created_at       INTEGER NOT NULL
);


-- ==========================================================================
--  LAYER C — RAZORPAY-NATIVE TRANSACTION ENTITIES
--  Field sets and enum values follow Razorpay's public API objects.
-- ==========================================================================

-- [RZP] Order entity.
CREATE TABLE orders (
    id               TEXT PRIMARY KEY,          -- order_xxxxxxxxxxxx
    merchant_id      TEXT    NOT NULL REFERENCES merchants(id),
    amount           INTEGER NOT NULL CHECK (amount >= 0),
    amount_paid      INTEGER NOT NULL DEFAULT 0,
    amount_due       INTEGER NOT NULL DEFAULT 0,
    currency         TEXT    NOT NULL DEFAULT 'INR',
    receipt          TEXT,                      -- merchant's internal reference
    status           TEXT    NOT NULL CHECK (status IN ('created','attempted','paid')),
    attempts         INTEGER NOT NULL DEFAULT 0,
    offer_id         TEXT,
    -- [EXT] Not a Razorpay Order field. Added because the training dataset
    -- carries a country per invoice and it is a usable feature; Razorpay
    -- itself exposes geography on the customer/address, not the order.
    country          TEXT,
    notes            TEXT,                      -- json
    created_at       INTEGER NOT NULL
);

-- [RZP] Payment entity. Field set follows Razorpay's documented Payment object.
CREATE TABLE payments (
    id               TEXT PRIMARY KEY,          -- pay_xxxxxxxxxxxx
    merchant_id      TEXT    NOT NULL REFERENCES merchants(id),
    order_id         TEXT             REFERENCES orders(id),
    customer_id      TEXT             REFERENCES customers(id),  -- resolved at ingest
    amount           INTEGER NOT NULL CHECK (amount >= 0),
    currency         TEXT    NOT NULL DEFAULT 'INR',
    status           TEXT    NOT NULL CHECK (status IN
                        ('created','authorized','captured','refunded','failed')),
    -- Razorpay's documented methods are card, netbanking, wallet, emi, upi.
    -- 'cod' is OUR extension: not a Razorpay method, but a first-order driver
    -- of return risk, so it is admitted here and flagged as non-native.
    method           TEXT    NOT NULL CHECK (method IN
                        ('card','netbanking','wallet','emi','upi','cod')),
    captured         BOOLEAN NOT NULL DEFAULT 0,
    international    BOOLEAN NOT NULL DEFAULT 0,
    description      TEXT,
    -- method-specific identifiers, populated according to `method`
    card_id          TEXT,
    bank             TEXT,
    wallet           TEXT,
    vpa              TEXT,
    acquirer_data    TEXT,                      -- json
    -- payer contact
    email            TEXT,
    contact          TEXT,
    -- Razorpay's commercials
    fee              INTEGER NOT NULL DEFAULT 0,
    tax              INTEGER NOT NULL DEFAULT 0,
    -- failure detail
    error_code       TEXT,
    error_description TEXT,
    error_source     TEXT,
    error_step       TEXT,
    error_reason     TEXT,
    -- refund rollup
    amount_refunded  INTEGER NOT NULL DEFAULT 0,
    refund_status    TEXT    CHECK (refund_status IS NULL
                                    OR refund_status IN ('partial','full')),
    notes            TEXT,                      -- json
    created_at       INTEGER NOT NULL
);

-- [RZP] Refund entity, plus one extension column.
CREATE TABLE refunds (
    id               TEXT PRIMARY KEY,          -- rfnd_xxxxxxxxxxxx
    merchant_id      TEXT    NOT NULL REFERENCES merchants(id),
    payment_id       TEXT    NOT NULL REFERENCES payments(id),
    amount           INTEGER NOT NULL CHECK (amount >= 0),
    currency         TEXT    NOT NULL DEFAULT 'INR',
    status           TEXT    NOT NULL CHECK (status IN ('pending','processed','failed')),
    speed_requested  TEXT    CHECK (speed_requested IS NULL
                                    OR speed_requested IN ('normal','optimum')),
    speed_processed  TEXT    CHECK (speed_processed IS NULL
                                    OR speed_processed IN ('normal','instant')),
    receipt          TEXT,
    batch_id         TEXT,
    notes            TEXT,                      -- json
    -- [EXT] Razorpay carries no refund reason, but the distinction is decisive:
    -- a merchant-side refund (out of stock, pricing error) is NOT a return-risk
    -- event and must not enter the positive class. Set at ingest or via
    -- POST /outcomes.
    refund_reason    TEXT    CHECK (refund_reason IS NULL OR refund_reason IN
                        ('customer_return',        -- counts toward label_refunded
                         'customer_cancellation',  -- counts
                         'merchant_stock_out',     -- excluded
                         'merchant_error',         -- excluded
                         'duplicate_charge',       -- excluded
                         'goodwill',               -- excluded
                         'other')),
    created_at       INTEGER NOT NULL
);

-- [RZP] Dispute entity ("chargeback"). Enum values follow Razorpay exactly.
CREATE TABLE disputes (
    id               TEXT PRIMARY KEY,          -- disp_xxxxxxxxxxxx
    merchant_id      TEXT    NOT NULL REFERENCES merchants(id),
    payment_id       TEXT    NOT NULL REFERENCES payments(id),
    amount           INTEGER NOT NULL CHECK (amount >= 0),
    currency         TEXT    NOT NULL DEFAULT 'INR',
    -- amount actually removed from the merchant balance when a dispute is lost;
    -- this, not `amount`, is the true false-negative cost input.
    amount_deducted  INTEGER NOT NULL DEFAULT 0,
    reason_code      TEXT    NOT NULL,
    reason_description TEXT,
    respond_by       INTEGER,                   -- epoch deadline for evidence
    status           TEXT    NOT NULL CHECK (status IN
                        ('open','under_review','won','lost','closed')),
    -- 'fraud' and 'retrieval' are Razorpay phases that are OUT OF SCOPE for
    -- this project: fraud is third-party identity theft (a different track),
    -- retrieval is an information request, not a loss. They are stored so they
    -- can be excluded explicitly rather than silently absorbed into the label.
    phase            TEXT    NOT NULL CHECK (phase IN
                        ('fraud','retrieval','chargeback','pre_arbitration','arbitration')),
    evidence         TEXT,                      -- json
    created_at       INTEGER NOT NULL
);


-- ==========================================================================
--  LAYER D — COMMERCE DETAIL
--  Product-level and velocity features live here.
-- ==========================================================================

-- [EXT] Product catalogue. Backs the product-level feature family:
-- category baseline return rate, price relative to category average,
-- size/fit dependency.
CREATE TABLE products (
    id               TEXT PRIMARY KEY,          -- prod_xxxxxxxxxxxx
    merchant_id      TEXT    NOT NULL REFERENCES merchants(id),
    sku              TEXT    NOT NULL,
    name             TEXT,
    category         TEXT    NOT NULL,
    subcategory      TEXT,
    list_price       INTEGER NOT NULL CHECK (list_price >= 0),
    -- how strongly the item's fit drives returns; apparel/footwear dominate
    size_variant_type TEXT   NOT NULL DEFAULT 'none'
                        CHECK (size_variant_type IN ('none','apparel','footwear','other')),
    created_at       INTEGER NOT NULL,
    UNIQUE (merchant_id, sku)
);

-- [EXT] Line items on an order.
CREATE TABLE order_items (
    id               TEXT PRIMARY KEY,          -- oitem_xxxxxxxxxxxx
    order_id         TEXT    NOT NULL REFERENCES orders(id),
    product_id       TEXT             REFERENCES products(id),
    sku              TEXT    NOT NULL,
    category         TEXT,                      -- denormalised for cold-start
    unit_price       INTEGER NOT NULL CHECK (unit_price >= 0),
    quantity         INTEGER NOT NULL CHECK (quantity > 0),
    size_variant     TEXT,
    discount_applied INTEGER NOT NULL DEFAULT 0
);

-- [EXT] Per-transaction context for velocity features only.
-- SCOPE GUARD: these columns support "how many attempts from this device in
-- the last hour" (a per-transaction velocity signal). They must NOT be joined
-- across accounts to build identity graphs — that is the abuse-ring direction,
-- which this project explicitly does not build.
CREATE TABLE payment_context (
    payment_id       TEXT PRIMARY KEY REFERENCES payments(id),
    ip               TEXT,
    device_id        TEXT,
    session_id       TEXT,
    user_agent       TEXT,
    billing_pincode  TEXT,
    shipping_pincode TEXT,
    address_mismatch BOOLEAN,
    created_at       INTEGER NOT NULL
);


-- ==========================================================================
--  LAYER E — SERVING-TIME AGGREGATES
--  Mutable cache. Never read at training time. See the leakage rule below.
-- ==========================================================================

-- [EXT] SERVING-TIME AGGREGATE CACHE — NOT A TRAINING SOURCE.
--
-- This table holds one mutable row per customer, refreshed as outcomes land.
-- `as_of_ts` records the moment the counters are valid for, and exists so that
-- the table can never be mistaken for a timeless fact.
--
-- LEAKAGE RULE: training and evaluation MUST NOT read this table. A row read
-- at training time carries the customer's FINAL return rate, which already
-- contains the outcome of the very payment being predicted plus every later
-- outcome. A chronological split on payments.created_at does not catch this,
-- because the leak arrives through an aggregate that has no time dimension.
-- Training reads risk_score_features (frozen at scoring time) or recomputes
-- as-of values from refunds/disputes filtered to created_at < the payment's
-- created_at. See the reference query at the foot of this file.
CREATE TABLE customer_stats_current (
    customer_id      TEXT PRIMARY KEY REFERENCES customers(id),
    merchant_id      TEXT    NOT NULL REFERENCES merchants(id),
    as_of_ts         INTEGER NOT NULL,          -- counters are true as of this epoch
    total_orders             INTEGER NOT NULL DEFAULT 0,
    total_payments_captured  INTEGER NOT NULL DEFAULT 0,
    total_refunds            INTEGER NOT NULL DEFAULT 0,
    total_returns            INTEGER NOT NULL DEFAULT 0,  -- refunds that are returns
    total_disputes           INTEGER NOT NULL DEFAULT 0,  -- in-scope phases only
    -- denominators are captured payments, not orders: disputes attach to
    -- payments, and one order can carry several attempts.
    past_return_rate         REAL    NOT NULL DEFAULT 0,
    past_dispute_rate        REAL    NOT NULL DEFAULT 0,
    computed_at      INTEGER NOT NULL
);


-- ==========================================================================
--  LAYER F — LABELS & DATASETS
--  The training target, with maturity and exclusions made explicit.
-- ==========================================================================

-- [EXT] The training target, with maturity and exclusions made explicit.
--
-- Two things v1 could not express and this table now can:
--   * MATURITY — a payment with no dispute row is not "clean" until its
--     dispute window has closed. Without is_mature, a model trained on recent
--     data learns that recent orders are safe, and held-out metrics flatter.
--   * EXCLUSIONS — fraud-phase disputes, retrieval requests and merchant-side
--     refunds are not return/chargeback risk. They are recorded and excluded,
--     not silently counted as positives.
CREATE TABLE risk_labels (
    payment_id       TEXT PRIMARY KEY REFERENCES payments(id),
    merchant_id      TEXT    NOT NULL REFERENCES merchants(id),
    label_refunded   BOOLEAN NOT NULL DEFAULT 0,  -- customer-driven refunds only
    label_disputed   BOOLEAN NOT NULL DEFAULT 0,  -- in-scope dispute phases only
    label_risk       BOOLEAN NOT NULL DEFAULT 0,  -- refunded OR disputed
    outcome_observed_at INTEGER,                  -- when the outcome landed, if any
    label_matured_at INTEGER NOT NULL,            -- payment.created_at + window
    is_mature        BOOLEAN NOT NULL DEFAULT 0,  -- label_matured_at <= now
    excluded_reason  TEXT    CHECK (excluded_reason IS NULL OR excluded_reason IN
                        ('fraud_dispute','retrieval_request','merchant_refund',
                         'immature','payment_not_captured')),
    label_version    INTEGER NOT NULL DEFAULT 1,
    derived_at       INTEGER NOT NULL
);

-- [EXT] A frozen train/test split. Written once, hashed, and referenced by
-- every model and evaluation, so a re-run can never silently re-split.
CREATE TABLE datasets (
    id               TEXT PRIMARY KEY,          -- ds_xxxxxxxxxxxx
    merchant_id      TEXT    NOT NULL REFERENCES merchants(id),
    name             TEXT    NOT NULL,
    train_until      INTEGER NOT NULL,          -- chronological cut point, epoch
    exclude_immature BOOLEAN NOT NULL DEFAULT 1,
    n_train          INTEGER NOT NULL DEFAULT 0,
    n_test           INTEGER NOT NULL DEFAULT 0,
    positive_rate    REAL,
    split_hash       TEXT    NOT NULL,          -- hash of the member set
    created_at       INTEGER NOT NULL
);

CREATE TABLE dataset_members (
    dataset_id       TEXT    NOT NULL REFERENCES datasets(id),
    payment_id       TEXT    NOT NULL REFERENCES payments(id),
    split            TEXT    NOT NULL CHECK (split IN ('train','test')),
    PRIMARY KEY (dataset_id, payment_id)
);


-- ==========================================================================
--  LAYER G — MODELS, COST MODEL & EVALUATION
-- ==========================================================================

CREATE TABLE models (
    id               TEXT PRIMARY KEY,          -- mdl_xxxxxxxxxxxx
    merchant_id      TEXT    NOT NULL REFERENCES merchants(id),
    algorithm        TEXT    NOT NULL CHECK (algorithm IN
                        ('logistic_regression','lightgbm','xgboost')),
    dataset_id       TEXT    NOT NULL REFERENCES datasets(id),
    hyperparameters  TEXT,                      -- json
    feature_list     TEXT    NOT NULL,          -- json array, ordered
    train_until      INTEGER NOT NULL,
    status           TEXT    NOT NULL CHECK (status IN
                        ('training','trained','active','archived','failed')),
    artifact_path    TEXT,                      -- joblib/pickle location
    trained_at       INTEGER,
    activated_at     INTEGER,
    created_at       INTEGER NOT NULL
);

CREATE TABLE model_feature_importance (
    model_id         TEXT    NOT NULL REFERENCES models(id),
    feature_name     TEXT    NOT NULL,
    importance       REAL    NOT NULL,          -- coefficient or gain
    rank             INTEGER NOT NULL,
    PRIMARY KEY (model_id, feature_name)
);

-- [EXT] False-positive and false-negative costs, stored as the formula inputs
-- rather than a bare number, so the figure can be audited instead of trusted.
CREATE TABLE cost_model_config (
    id               TEXT PRIMARY KEY,          -- cost_xxxxxxxxxxxx
    merchant_id      TEXT    NOT NULL REFERENCES merchants(id),
    currency         TEXT    NOT NULL DEFAULT 'INR',
    -- false positive: a genuine customer is held or friction'd
    lost_sale_probability REAL    NOT NULL,
    average_order_value   INTEGER NOT NULL,
    trust_penalty         INTEGER NOT NULL,
    cost_per_fp           INTEGER NOT NULL,     -- p*AOV + trust_penalty
    -- false negative: a real return or chargeback is missed
    refunded_amount       INTEGER NOT NULL,
    chargeback_fee        INTEGER NOT NULL,
    logistics_restocking  INTEGER NOT NULL,
    cost_per_fn           INTEGER NOT NULL,
    reason           TEXT    NOT NULL,          -- why this version exists
    created_by       TEXT,
    is_current       BOOLEAN NOT NULL DEFAULT 0,
    created_at       INTEGER NOT NULL
);

-- [EXT] One held-out evaluation run. Records the cost model in force at run
-- time, so an old run can never be reinterpreted under new cost assumptions.
CREATE TABLE evaluations (
    id               TEXT PRIMARY KEY,          -- eval_xxxxxxxxxxxx
    merchant_id      TEXT    NOT NULL REFERENCES merchants(id),
    model_id         TEXT    NOT NULL REFERENCES models(id),
    dataset_id       TEXT    NOT NULL REFERENCES datasets(id),
    cost_model_id    TEXT    NOT NULL REFERENCES cost_model_config(id),
    n_samples        INTEGER,
    base_rate        REAL,                      -- positive rate in the test slice
    threshold_evaluated REAL,
    precision_score  REAL,
    recall_score     REAL,
    f1_score         REAL,
    pr_auc           REAL,
    roc_auc          REAL,
    tp INTEGER, fp INTEGER, fn INTEGER, tn INTEGER,
    total_cost       INTEGER,
    optimal_threshold REAL,
    status           TEXT    NOT NULL CHECK (status IN ('running','completed','failed')),
    started_at       INTEGER NOT NULL,
    completed_at     INTEGER
);

-- [EXT] The threshold sweep: one row per threshold step. This is the series
-- behind the cost-vs-threshold chart, and evaluations.optimal_threshold is the
-- argmin of total_cost over these rows — not the accuracy-maximising point.
CREATE TABLE evaluation_points (
    eval_id          TEXT    NOT NULL REFERENCES evaluations(id),
    threshold        REAL    NOT NULL,
    tp INTEGER NOT NULL, fp INTEGER NOT NULL,
    fn INTEGER NOT NULL, tn INTEGER NOT NULL,
    precision_score  REAL,
    recall_score     REAL,
    total_cost       INTEGER NOT NULL,
    PRIMARY KEY (eval_id, threshold)
);


-- ==========================================================================
--  LAYER H — THRESHOLD CONFIGURATION
--  Versioned; links back to the evaluation run that produced the cut points.
-- ==========================================================================

-- [EXT] Band cut points. source='optimized' links back to the evaluation run
-- whose cost curve produced them.
CREATE TABLE threshold_config (
    id               TEXT PRIMARY KEY,          -- thr_xxxxxxxxxxxx
    merchant_id      TEXT    NOT NULL REFERENCES merchants(id),
    threshold_low    REAL    NOT NULL CHECK (threshold_low BETWEEN 0 AND 1),
    threshold_high   REAL    NOT NULL CHECK (threshold_high BETWEEN 0 AND 1),
    source           TEXT    NOT NULL CHECK (source IN ('manual','optimized')),
    eval_id          TEXT             REFERENCES evaluations(id),
    reason           TEXT    NOT NULL,
    created_by       TEXT,
    is_current       BOOLEAN NOT NULL DEFAULT 0,
    created_at       INTEGER NOT NULL,
    CHECK (threshold_low <= threshold_high)
);


-- ==========================================================================
--  LAYER I — SCORING
-- ==========================================================================

-- [EXT] One row per score produced. Backs POST /risk/score and /risk/scores/*.
CREATE TABLE risk_scores (
    id               TEXT PRIMARY KEY,          -- score_xxxxxxxxxxxx
    merchant_id      TEXT    NOT NULL REFERENCES merchants(id),
    payment_id       TEXT             REFERENCES payments(id),
    order_id         TEXT             REFERENCES orders(id),
    customer_id      TEXT             REFERENCES customers(id),
    model_id         TEXT    NOT NULL REFERENCES models(id),
    risk_probability REAL    NOT NULL CHECK (risk_probability BETWEEN 0 AND 1),
    risk_band        TEXT    NOT NULL CHECK (risk_band IN ('low','medium','high')),
    threshold_applied REAL   NOT NULL,
    threshold_config_id TEXT NOT NULL REFERENCES threshold_config(id),
    recommendation   TEXT    NOT NULL CHECK (recommendation IN
                        ('allow','manual_review','hold_payout','request_verification')),
    customer_history TEXT    NOT NULL CHECK (customer_history IN ('present','none')),
    review_id        TEXT,                      -- set when the score opens a review
    scoring_latency_ms INTEGER,
    created_at       INTEGER NOT NULL
);

-- [EXT] THE AS-OF FEATURE SNAPSHOT.
--
-- The exact feature vector the model saw, frozen at scoring time, with each
-- feature's contribution to the score. This table does three jobs at once:
--   1. makes a score reproducible months later, when the aggregates have moved;
--   2. supplies GET /risk/scores/{id}/features;
--   3. is the leakage-free training source — every value here was computed
--      from data that existed strictly before the transaction.
CREATE TABLE risk_score_features (
    score_id         TEXT    NOT NULL REFERENCES risk_scores(id),
    feature_name     TEXT    NOT NULL,
    feature_value    REAL,
    contribution     REAL,                      -- signed contribution to the score
    PRIMARY KEY (score_id, feature_name)
);

-- [EXT] LLM output. Read-only with respect to the score: there is no column
-- here that can alter risk_probability, risk_band or recommendation.
-- risk_probability_at_generation copies the score the text was written against,
-- so a stale explanation can be detected rather than trusted.
CREATE TABLE risk_explanations (
    score_id         TEXT PRIMARY KEY REFERENCES risk_scores(id),
    status           TEXT    NOT NULL CHECK (status IN ('pending','ready','failed')),
    summary          TEXT,
    risk_probability_at_generation REAL,
    generated_by     TEXT,                      -- model identifier
    prompt_version   TEXT,
    error_message    TEXT,
    requested_at     INTEGER NOT NULL,
    generated_at     INTEGER
);


-- ==========================================================================
--  LAYER J — HUMAN REVIEW
--  The only path by which a score becomes an action.
-- ==========================================================================

-- [EXT] People who may decide. A decision without a row here is impossible.
CREATE TABLE reviewers (
    id               TEXT PRIMARY KEY,          -- usr_xxxxxxxxxxxx
    merchant_id      TEXT    NOT NULL REFERENCES merchants(id),
    name             TEXT    NOT NULL,
    email            TEXT    NOT NULL,
    active           BOOLEAN NOT NULL DEFAULT 1,
    created_at       INTEGER NOT NULL
);

-- [EXT] Review queue item, opened when a score lands in a flagged band.
CREATE TABLE reviews (
    id               TEXT PRIMARY KEY,          -- rev_xxxxxxxxxxxx
    merchant_id      TEXT    NOT NULL REFERENCES merchants(id),
    score_id         TEXT    NOT NULL REFERENCES risk_scores(id),
    payment_id       TEXT             REFERENCES payments(id),
    status           TEXT    NOT NULL CHECK (status IN ('open','assigned','closed')),
    assignee_id      TEXT             REFERENCES reviewers(id),
    opened_at        INTEGER NOT NULL,
    assigned_at      INTEGER,
    closed_at        INTEGER
);

-- [EXT] APPEND-ONLY decision log. A reversal is a new row on a reopened
-- review, never an update. reviewer_id is NOT NULL by design: there is no
-- system reviewer, and neither the model nor the LLM can write here.
CREATE TABLE review_decisions (
    id               TEXT PRIMARY KEY,
    review_id        TEXT    NOT NULL REFERENCES reviews(id),
    reviewer_id      TEXT    NOT NULL REFERENCES reviewers(id),
    decision         TEXT    NOT NULL CHECK (decision IN
                        ('approve','hold_payout','request_verification','cancel_order')),
    reason           TEXT,
    agreed_with_model BOOLEAN,
    created_at       INTEGER NOT NULL
);

CREATE TABLE review_notes (
    id               TEXT PRIMARY KEY,
    review_id        TEXT    NOT NULL REFERENCES reviews(id),
    reviewer_id      TEXT    NOT NULL REFERENCES reviewers(id),
    note             TEXT    NOT NULL,
    created_at       INTEGER NOT NULL
);


-- ==========================================================================
--  LAYER K — OPERATIONS
-- ==========================================================================

CREATE TABLE ingest_jobs (
    id               TEXT PRIMARY KEY,          -- job_xxxxxxxxxxxx
    merchant_id      TEXT    NOT NULL REFERENCES merchants(id),
    file_name        TEXT,
    entity_type      TEXT    CHECK (entity_type IN
                        ('orders','payments','refunds','disputes','products','mixed')),
    status           TEXT    NOT NULL CHECK (status IN
                        ('queued','running','completed','failed')),
    rows_total       INTEGER NOT NULL DEFAULT 0,
    rows_accepted    INTEGER NOT NULL DEFAULT 0,
    rows_rejected    INTEGER NOT NULL DEFAULT 0,
    rejections       TEXT,                      -- json, per-row reasons
    started_at       INTEGER NOT NULL,
    completed_at     INTEGER
);

-- [EXT] Inbound Razorpay webhooks. The primary key is the provider's event id,
-- which makes redelivery idempotent for free.
CREATE TABLE webhook_events (
    id               TEXT PRIMARY KEY,          -- provider event id
    merchant_id      TEXT    NOT NULL REFERENCES merchants(id),
    event_type       TEXT    NOT NULL,          -- payment.captured, refund.created, ...
    payload          TEXT    NOT NULL,          -- json
    signature_verified BOOLEAN NOT NULL DEFAULT 0,
    processing_status TEXT   NOT NULL CHECK (processing_status IN
                        ('received','processed','failed','ignored')),
    error_message    TEXT,
    received_at      INTEGER NOT NULL,
    processed_at     INTEGER
);

-- [EXT] Backs the Idempotency-Key header on every state-changing POST.
CREATE TABLE idempotency_keys (
    key              TEXT    NOT NULL,
    merchant_id      TEXT    NOT NULL REFERENCES merchants(id),
    endpoint         TEXT    NOT NULL,
    request_hash     TEXT    NOT NULL,
    response_status  INTEGER,
    response_body    TEXT,                      -- json
    created_at       INTEGER NOT NULL,
    expires_at       INTEGER NOT NULL,          -- created_at + 24h
    PRIMARY KEY (merchant_id, key)
);


-- ============================================================================
--  INDEXES
--  Without these the velocity features degrade to full table scans: they are
--  per-customer range queries over created_at.
-- ============================================================================

CREATE INDEX idx_orders_merchant_created    ON orders (merchant_id, created_at);
CREATE INDEX idx_orders_receipt             ON orders (merchant_id, receipt);

CREATE INDEX idx_payments_order             ON payments (order_id);
CREATE INDEX idx_payments_customer_created  ON payments (customer_id, created_at);
CREATE INDEX idx_payments_merchant_created  ON payments (merchant_id, created_at);
CREATE INDEX idx_payments_email             ON payments (merchant_id, email);
CREATE INDEX idx_payments_contact           ON payments (merchant_id, contact);
CREATE INDEX idx_payments_status_method     ON payments (merchant_id, status, method);

CREATE INDEX idx_refunds_payment            ON refunds (payment_id);
CREATE INDEX idx_refunds_created            ON refunds (merchant_id, created_at);

CREATE INDEX idx_disputes_payment           ON disputes (payment_id);
CREATE INDEX idx_disputes_created           ON disputes (merchant_id, created_at);
CREATE INDEX idx_disputes_phase_status      ON disputes (merchant_id, phase, status);

CREATE INDEX idx_order_items_order          ON order_items (order_id);
CREATE INDEX idx_order_items_product        ON order_items (product_id);
CREATE INDEX idx_products_category          ON products (merchant_id, category);

CREATE INDEX idx_ctx_device                 ON payment_context (device_id, created_at);
CREATE INDEX idx_ctx_ip                     ON payment_context (ip, created_at);

CREATE INDEX idx_customers_email            ON customers (merchant_id, email);
CREATE INDEX idx_customers_contact          ON customers (merchant_id, contact);

CREATE INDEX idx_scores_payment             ON risk_scores (payment_id, created_at);
CREATE INDEX idx_scores_merchant_created    ON risk_scores (merchant_id, created_at);
CREATE INDEX idx_scores_band                ON risk_scores (merchant_id, risk_band, created_at);
CREATE INDEX idx_scores_model               ON risk_scores (model_id);

CREATE INDEX idx_reviews_status             ON reviews (merchant_id, status, opened_at);
CREATE INDEX idx_reviews_assignee           ON reviews (assignee_id, status);
CREATE INDEX idx_review_decisions_review    ON review_decisions (review_id, created_at);

CREATE INDEX idx_labels_risk                ON risk_labels (merchant_id, label_risk, is_mature);
CREATE INDEX idx_labels_matured             ON risk_labels (label_matured_at);

CREATE INDEX idx_dataset_members_split      ON dataset_members (dataset_id, split);
CREATE INDEX idx_eval_points_cost           ON evaluation_points (eval_id, total_cost);
CREATE INDEX idx_webhook_status             ON webhook_events (merchant_id, processing_status);
CREATE INDEX idx_idem_expiry                ON idempotency_keys (expires_at);

-- Exactly one active model per merchant.
CREATE UNIQUE INDEX idx_models_one_active
    ON models (merchant_id) WHERE status = 'active';

-- Exactly one current config row per merchant, per config type.
CREATE UNIQUE INDEX idx_costmodel_one_current
    ON cost_model_config (merchant_id) WHERE is_current = 1;
CREATE UNIQUE INDEX idx_threshold_one_current
    ON threshold_config (merchant_id) WHERE is_current = 1;


-- ============================================================================
--  REFERENCE QUERY 1 — LABEL DERIVATION
--  Backs POST /risk-labels/rebuild. Note the two exclusions that v1 could not
--  express: fraud/retrieval dispute phases, and merchant-side refunds.
-- ============================================================================

-- INSERT INTO risk_labels (...)
SELECT
    p.id                                    AS payment_id,
    p.merchant_id,
    -- customer-driven refunds only
    CASE WHEN EXISTS (
        SELECT 1 FROM refunds r
        WHERE r.payment_id = p.id
          AND r.status = 'processed'
          AND r.refund_reason IN ('customer_return','customer_cancellation')
    ) THEN 1 ELSE 0 END                     AS label_refunded,
    -- in-scope dispute phases only: fraud and retrieval are excluded
    CASE WHEN EXISTS (
        SELECT 1 FROM disputes d
        WHERE d.payment_id = p.id
          AND d.phase NOT IN ('fraud','retrieval')
          AND d.status IN ('open','under_review','lost','closed')
    ) THEN 1 ELSE 0 END                     AS label_disputed,
    p.created_at + (m.dispute_window_days * 86400) AS label_matured_at,
    CASE WHEN p.created_at + (m.dispute_window_days * 86400) <= strftime('%s','now')
         THEN 1 ELSE 0 END                  AS is_mature
FROM payments p
JOIN merchants m ON m.id = p.merchant_id
WHERE p.captured = 1;


-- ============================================================================
--  REFERENCE QUERY 2 — LEAKAGE-FREE AS-OF CUSTOMER FEATURES
--  How customer history is computed for TRAINING. Every aggregate is bounded
--  by the transaction's own created_at, so no future outcome can enter the
--  feature. Contrast with reading customer_stats_current, which would hand the
--  model the customer's final rates — including the outcome being predicted.
-- ============================================================================

SELECT
    p.id                                    AS payment_id,
    p.created_at                            AS as_of_ts,
    COUNT(prior.id)                         AS prior_payments,
    -- prior returns, strictly before this transaction
    SUM(CASE WHEN EXISTS (
        SELECT 1 FROM refunds r
        WHERE r.payment_id = prior.id
          AND r.created_at < p.created_at
          AND r.refund_reason IN ('customer_return','customer_cancellation')
    ) THEN 1 ELSE 0 END)                    AS prior_returns,
    -- prior in-scope disputes, strictly before this transaction
    SUM(CASE WHEN EXISTS (
        SELECT 1 FROM disputes d
        WHERE d.payment_id = prior.id
          AND d.created_at < p.created_at
          AND d.phase NOT IN ('fraud','retrieval')
    ) THEN 1 ELSE 0 END)                    AS prior_disputes
FROM payments p
LEFT JOIN payments prior
       ON prior.customer_id = p.customer_id
      AND prior.merchant_id = p.merchant_id
      AND prior.created_at  < p.created_at      -- the whole leakage fix is here
      AND prior.captured    = 1
GROUP BY p.id, p.created_at;
