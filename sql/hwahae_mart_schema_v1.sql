-- ============================================================
-- Hwahae K-Beauty Sentiment Analysis — Analytics Mart Layer (v1)
-- Database: hwahae_project (same DB as hwahae_schema.sql)
-- Sits downstream of the normalized layer (hwahae_schema.sql) and
-- the preprocessing layer (mysql_preprocessing.sql):
--
--   Hwahae reviews (collected)
--     -> MySQL normalized layer          (hwahae_schema.sql)
--     -> MySQL preprocessing layer       (mysql_preprocessing.sql)
--     -> KcELECTRA sentiment classification (Python / Kaggle, offline)
--     -> MySQL ANALYTICS MART LAYER  <---  this file
--     -> BI dashboard / LLM report
--
-- Rules implemented here:
--   - review-count-per-month gate: 20
--   - decline trigger: CUSUM-based, see object #4 below
--   - chronic low-confidence flag: >= 30% of trailing months failing the
--     gate (lives in product_confidence_flags_v1)
--   - minimum history: >= 4 months before decline-eligible, expressed as
--     3 baseline months + the current one; the baseline months do NOT
--     need to be consecutive calendar months — see object #4 below
--   - category comparison gate: n >= 3 products (NOT implemented here —
--     it's a plain COUNT(DISTINCT product_id) ... HAVING clause against
--     products_v3 at query time, doesn't need a stored object)
--   - risk score: single-factor (correlational, not causal; no
--     multi-factor weighted composite), computed as a CUSUM cumulative-sum
--     statistic, see object #4 below. A second, independent,
--     informational-only signal (notable_month_dip) sits alongside it.
--   - complaint extraction: BERTopic (bridge tables only — the
--     clustering itself runs in Python, same pattern as the
--     sentiment-scoring bridge table below)
--
-- Design note — table vs. view: the objects that require an external
-- Python/Kaggle batch job to populate are physical TABLEs:
-- review_sentiment_scores_v1, complaint_topics_v1 +
-- product_complaint_theme_mentions_v1, and voc_decline_risk_v1 (object #4
-- below). Their reasons differ: review_sentiment_scores_v1
-- and the BERTopic bridge tables MUST be physical tables because the
-- computation (KcELECTRA inference, BERTopic clustering) literally cannot
-- run inside MySQL. voc_decline_risk_v1's CUSUM trigger has recursive
-- state (S_t depends on S_(t-1)) that IS computable as a recursive CTE —
-- product_confidence_flags_v1 above already does per-product recursion
-- for exactly this reason — but it is computed in Python
-- (populate_voc_decline_risk.py) instead, for testability with no
-- live-DB access. See object #4's own comment block. Everything ELSE
-- derivable by pure SQL from the bridge tables remains a VIEW, computed
-- live — same non-destructive, no-stale-refresh reasoning already used for
-- hwahae_reviews_clean_v3 in mysql_preprocessing.sql. Row counts here
-- (thousands, not hundreds of thousands) are small enough that live
-- computation is not a performance concern the way the 814K-row
-- reviews_clean view is.
--
-- Versioning: mart objects use their own _v1 suffix, independent of the
-- normalized layer's _v3 (that version tracks scrape-schema changes;
-- this one tracks mart-schema changes, a separate concern).
-- ============================================================

USE `hwahae_project`;


-- ============================================================
-- 1. review_sentiment_scores_v1 — bridge table (physical)
--    One row per review. Populated by loading the KcELECTRA
--    inference output (from the Kaggle checkpoints) back into MySQL — this cannot be a view, since
--    scoring happens outside the database.
--
--    Review-level model: 3-class (frustrated / neutral_mixed /
--    high_satisfaction) -> maps directly to (negative / neutral /
--    positive). review_net_sentiment = P(positive) - P(negative).
--
--    Segment-level model: 4-class (positive_benefit / minor_complaint /
--    strong_complaint / usage_tip) — NOT the same 3-class scheme as
--    review-level (pros
--    is ~100% positive_benefit and tip ~100% usage_tip by construction,
--    since those labels came from the segment field itself, not judged
--    sentiment; the real signal is entirely within cons, which splits
--    minor_complaint/strong_complaint by rating). Mapped to a -1..+1
--    net-sentiment scale via a fixed severity weighting, not a raw
--    P(pos)-P(neg) (that pair doesn't exist in this label space):
--        segment_net_sentiment = p_positive_benefit
--                                 - 0.5 * p_minor_complaint
--                                 - 1.0 * p_strong_complaint
--    (p_usage_tip contributes 0 — instructional text is treated as
--    sentiment-neutral, not positive or negative.) The 0.5/1.0 weights
--    mirror the same severity split already baked into the training
--    labels (rating 1-2 -> strong, rating 4-5 -> minor), so they're not
--    arbitrary — they're the model's own label-severity ordering made
--    explicit as a numeric scale. Documented as a heuristic mapping,
--    same "no ground truth to validate against" caveat as the risk score.
-- ============================================================

CREATE TABLE IF NOT EXISTS review_sentiment_scores_v1 (
    review_id                  BIGINT PRIMARY KEY,
    product_id                 BIGINT NOT NULL,
    review_month               DATE NOT NULL,     -- first-of-month, denormalized for GROUP BY

    -- Review-level (3-class: negative / neutral / positive)
    review_p_negative          DECIMAL(6,5) NOT NULL,
    review_p_neutral           DECIMAL(6,5) NOT NULL,
    review_p_positive          DECIMAL(6,5) NOT NULL,
    review_net_sentiment       DECIMAL(6,5)
        GENERATED ALWAYS AS (review_p_positive - review_p_negative) STORED,

    -- Segment-level (4-class: positive_benefit / minor_complaint /
    -- strong_complaint / usage_tip). NULL across a segment's four
    -- columns means that segment was empty on this review (e.g. no tip
    -- text) — not scored, not zero.
    pros_p_positive_benefit     DECIMAL(6,5) NULL,
    pros_p_minor_complaint      DECIMAL(6,5) NULL,
    pros_p_strong_complaint     DECIMAL(6,5) NULL,
    pros_p_usage_tip            DECIMAL(6,5) NULL,
    pros_net_sentiment          DECIMAL(6,5)
        GENERATED ALWAYS AS (
            pros_p_positive_benefit - 0.5 * pros_p_minor_complaint - pros_p_strong_complaint
        ) STORED,

    cons_p_positive_benefit     DECIMAL(6,5) NULL,
    cons_p_minor_complaint      DECIMAL(6,5) NULL,
    cons_p_strong_complaint     DECIMAL(6,5) NULL,
    cons_p_usage_tip            DECIMAL(6,5) NULL,
    cons_net_sentiment          DECIMAL(6,5)
        GENERATED ALWAYS AS (
            cons_p_positive_benefit - 0.5 * cons_p_minor_complaint - cons_p_strong_complaint
        ) STORED,

    tip_p_positive_benefit      DECIMAL(6,5) NULL,
    tip_p_minor_complaint       DECIMAL(6,5) NULL,
    tip_p_strong_complaint      DECIMAL(6,5) NULL,
    tip_p_usage_tip              DECIMAL(6,5) NULL,
    tip_net_sentiment           DECIMAL(6,5)
        GENERATED ALWAYS AS (
            tip_p_positive_benefit - 0.5 * tip_p_minor_complaint - tip_p_strong_complaint
        ) STORED,

    -- cons segment flagged negative/mixed -> feeds BERTopic input
    cons_is_complaint          BOOLEAN
        GENERATED ALWAYS AS (cons_net_sentiment IS NOT NULL AND cons_net_sentiment < 0) STORED,

    -- VARCHAR(80), not 50: the full-corpus run's model_version value
    -- (encoding both heads' checkpoint provenance, since review-level and
    -- segment-level came from different checkpoint generations) is 51
    -- characters long; at VARCHAR(50) every row fails with MySQL 1406
    -- (Data too long).
    model_version               VARCHAR(80) NOT NULL,  -- e.g. 'review:retrained-20260806|segment:baseline-20260730'
    scored_at                    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_product_month (product_id, review_month),
    INDEX idx_cons_is_complaint (cons_is_complaint),

    CONSTRAINT fk_scores_review_v1
        FOREIGN KEY (review_id)
        REFERENCES hwahae_reviews_raw_v3(review_id)
        ON DELETE CASCADE,
    CONSTRAINT fk_scores_product_v1
        FOREIGN KEY (product_id)
        REFERENCES products_v3(product_id)
)
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;


-- ============================================================
-- 2. product_monthly_sentiment_v1 — mart table #1 (VIEW)
--    Monthly per-product sentiment aggregates.
--    Grain: product_id x review_month. Sparse by design — a
--    zero-review month simply has no row here, matching the
--    "insufficient recent data" treatment: the dashboard
--    generates its own 12-month calendar and LEFT JOINs against this
--    view, rather than this view manufacturing all-zero rows itself.
-- ============================================================

CREATE OR REPLACE VIEW product_monthly_sentiment_v1 AS
SELECT
    product_id,
    review_month,
    COUNT(*)                                   AS review_count,
    (COUNT(*) >= 20)                            AS meets_review_count_gate,
    ROUND(AVG(review_net_sentiment), 5)          AS net_sentiment,
    ROUND(AVG(pros_net_sentiment), 5)             AS pros_net_sentiment,
    ROUND(AVG(cons_net_sentiment), 5)              AS cons_net_sentiment,
    ROUND(AVG(tip_net_sentiment), 5)                AS tip_net_sentiment,
    SUM(cons_is_complaint)                           AS complaint_count
FROM review_sentiment_scores_v1
GROUP BY product_id, review_month;


-- ============================================================
-- 3. product_confidence_flags_v1 — mart support view (VIEW)
--    Product-level data-sufficiency flags, evaluated as-of every
--    calendar month in each product's history (not just the latest),
--    so historical report replay (the as-of-date simulation)
--    sees the correct flag state for any past month, not just today's.
--
--    Built on a generated calendar per product (via recursive CTE) —
--    NOT on the sparse monthly view directly — because "history" for
--    the chronic low-confidence check means calendar months elapsed,
--    including zero-review gap months (a gap month trivially fails
--    the review-count gate; skipping it would understate the failure
--    rate). Recursion depth per product is bounded by its total
--    calendar span (~76 months for the longest-running product here,
--    well under MySQL/MariaDB's default cte_max_recursion_depth=1000).
-- ============================================================

CREATE OR REPLACE VIEW product_confidence_flags_v1 AS
WITH RECURSIVE product_span AS (
    SELECT product_id, MIN(review_month) AS first_month, MAX(review_month) AS last_month
    FROM product_monthly_sentiment_v1
    GROUP BY product_id
),
calendar AS (
    SELECT product_id, first_month AS calendar_month, last_month
    FROM product_span

    UNION ALL

    SELECT product_id, DATE_ADD(calendar_month, INTERVAL 1 MONTH), last_month
    FROM calendar
    WHERE calendar_month < last_month
),
calendar_with_gate AS (
    SELECT
        c.product_id,
        c.calendar_month,
        COALESCE(m.meets_review_count_gate, FALSE) AS meets_gate
    FROM calendar c
    LEFT JOIN product_monthly_sentiment_v1 m
        ON m.product_id = c.product_id
       AND m.review_month = c.calendar_month
)
SELECT
    product_id,
    calendar_month AS as_of_month,
    COUNT(*) OVER (
        PARTITION BY product_id ORDER BY calendar_month
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS months_of_history,
    (
        COUNT(*) OVER (
            PARTITION BY product_id ORDER BY calendar_month
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) < 4
    ) AS insufficient_history,
    ROUND(
        SUM(CASE WHEN NOT meets_gate THEN 1 ELSE 0 END) OVER (
            PARTITION BY product_id ORDER BY calendar_month
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )
        / COUNT(*) OVER (
            PARTITION BY product_id ORDER BY calendar_month
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ),
        4
    ) AS pct_trailing_months_failing_gate,
    (
        SUM(CASE WHEN NOT meets_gate THEN 1 ELSE 0 END) OVER (
            PARTITION BY product_id ORDER BY calendar_month
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )
        / COUNT(*) OVER (
            PARTITION BY product_id ORDER BY calendar_month
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )
    ) >= 0.30 AS low_confidence_product
FROM calendar_with_gate;


-- ============================================================
-- 4. voc_decline_risk_v1 — mart table #2 (TABLE, not VIEW —
--    see "Design note — table vs. view" above)
--    Risk-flag table. Grain: product_id x
--    as_of_month. Every product-month that exists in
--    product_monthly_sentiment_v1 gets a row here, gate-passing or
--    not (net_sentiment is denormalized straight across); the
--    decision columns (baseline/cusum/flags) are only ever non-NULL
--    for gate-passing months with sufficient baseline history.
--
--    WHY NOT A STREAK TRIGGER: a 3-consecutive-month hard-reset streak
--    structurally suppresses its own signal on this corpus —
--    mean-reversion makes a second declining month ~2x less likely
--    right after a first one (z≈5.0), so a single bounce-back month
--    erases an entire real, ongoing decline's evidence.
--
--    DESIGN: a CUSUM (cumulative sum control chart) trigger,
--    tolerant of a single off-month without losing accumulated
--    evidence of a real, noisy decline:
--        S_t = max(0, S_(t-1) + (baseline_net_sentiment − net_sentiment_t − k))
--    k ≈ σ/2 ≈ 0.0392 (σ = 0.07841667478303109, the empirical monthly-
--    average SD over gate-passing product-months). baseline_net_sentiment
--    is a trailing 6-month rolling mean of gate-passing months — it
--    bridges calendar gaps, it does NOT require 6 consecutive calendar
--    months. Minimum 3 baseline months + the current one = 4, matching
--    the "≥4 months minimum history" rule. Two-tier
--    decision threshold h: 4σ ≈ 0.3137 (voc_decline_broad_flag, the
--    looser tier) and 5σ ≈ 0.3921 (voc_decline_flag, the tight/"at
--    risk" tier referenced throughout this project's docs) — both
--    flags derive from the SAME cusum_score; h only changes which side
--    of the line counts as flagged, it does not change S_t's own
--    trajectory. k=0.5σ / h=4σ,5σ is standard Page's-CUSUM/SPC practice
--    (Montgomery-style), not invented for this project — the rolling
--    baseline (vs. a fixed target) and the minimum-history gate are
--    this project's own adaptations.
--
--    A SECOND, independent, informational
--    signal — notable_month_dip — ships alongside the CUSUM trigger,
--    not in place of it. Single-month, volume-adjusted (2.0× each
--    pair's own combined SE, σ_review·√(1/n_cur + 1/n_prev), σ_review =
--    0.37752813558355647), requires exact calendar adjacency (does NOT
--    bridge gaps the way the CUSUM baseline does). The two signals are
--    deliberately built to notice different things: a sustained drift
--    vs. one sharp month.
--
--    ARCHITECTURE: batch-populated physical TABLE
--    (populate_voc_decline_risk.py), not a live-computed VIEW — see
--    "Design note — table vs. view" above for why, given recursive
--    CTEs ARE technically capable of this (product_confidence_flags_v1
--    above already does per-product recursion). Practical implication:
--    this table reflects whatever review_sentiment_scores_v1 looked
--    like as of populate_voc_decline_risk.py's last run, not the
--    current instant — re-run it after any upstream rescore.
-- ============================================================

CREATE TABLE IF NOT EXISTS voc_decline_risk_v1 (
    product_id                     BIGINT NOT NULL,
    as_of_month                    DATE NOT NULL,

    -- Denormalized from product_monthly_sentiment_v1. Populated for
    -- every row, gate-passing or not.
    net_sentiment                  DECIMAL(6,5) NULL,

    -- Denormalized from product_monthly_sentiment_v1 alongside
    -- net_sentiment (same "always populated in practice, NULL allowed
    -- defensively and watched by a QA guard rather than a NOT NULL
    -- constraint" pattern), specifically so every NULL below is
    -- self-explanatory without a join back to that view. gate=FALSE
    -- fully explains a gate-failing month's NULL baseline/cusum on its
    -- own; when it's TRUE, baseline_months_available explains the rest
    -- (insufficient prior history, not this month's own volume).
    meets_review_count_gate        BOOLEAN NULL,
    review_count                   SMALLINT UNSIGNED NULL,

    -- CUSUM state. NULL/0 for gate-failing months and for the first 3
    -- gate-passing months of a product's history (3 baseline months +
    -- the current one = the >= 4 months minimum-history rule).
    baseline_net_sentiment         DECIMAL(6,5) NULL,       -- trailing 6-month rolling mean feeding this month's S_t
    baseline_months_available      TINYINT UNSIGNED NULL,   -- how many gate-passing months fed the baseline (<=6)
    cusum_score                    DECIMAL(6,5) NULL,       -- S_t; always >= 0 by construction (max(0, ...))
    decline_since                  DATE NULL,               -- first month of the current unbroken S_t > 0 run; NULL when S_t = 0

    -- Two-tier CUSUM flags, both derived from the SAME cusum_score.
    voc_decline_broad_flag         BOOLEAN NOT NULL DEFAULT FALSE,
    voc_decline_flag               BOOLEAN NOT NULL DEFAULT FALSE,

    -- The second signal: informational, single-month, no
    -- persistence requirement — distinct from the CUSUM columns above.
    notable_month_dip              BOOLEAN NOT NULL DEFAULT FALSE,
    notable_month_dip_drop         DECIMAL(6,5) NULL,       -- prior month's net_sentiment minus this month's, when flagged
    notable_month_dip_se           DECIMAL(6,5) NULL,       -- the pair's own combined SE it was compared against, when flagged

    computed_at                    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- when populate_voc_decline_risk.py last wrote this row

    PRIMARY KEY (product_id, as_of_month),
    INDEX idx_voc_decline_flag (voc_decline_flag),
    INDEX idx_voc_decline_broad_flag (voc_decline_broad_flag),
    INDEX idx_notable_month_dip (notable_month_dip),

    CONSTRAINT fk_voc_decline_risk_product_v1
        FOREIGN KEY (product_id)
        REFERENCES products_v3(product_id)
        ON DELETE CASCADE
)
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;


-- ============================================================
-- 5. complaint_topics_v1 — mart table #3a (physical)
--    BERTopic-discovered themes (global, fit once across all products'
--    complaint segments — not per-product). Bridge table: clustering
--    runs in Python (jhgan/ko-sroberta-multitask embeddings + default
--    HDBSCAN).
--
--    is_excluded/exclusion_reason/theme_group/polarity_type/qa_note hold
--    the three per-topic labeling-pass decisions (exclude flag, rollup
--    group, polarity classification) plus a free-text QA note.
-- ============================================================

CREATE TABLE IF NOT EXISTS complaint_topics_v1 (
    topic_id            INT PRIMARY KEY,          -- BERTopic's own topic ID (-1 = outlier/noise topic)
    topic_label          VARCHAR(500) NOT NULL,     -- top c-TF-IDF keywords, joined
    keyword_json           JSON NULL,                 -- full keyword+weight list, for the report LLM prompt
    model_version            VARCHAR(50) NOT NULL,       -- e.g. 'bertopic-v1-2026-08-05'
    created_at                 TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- The topic-labeling pass produces three decisions per topic -- an exclude flag, a rollup group, and a polarity
    -- classification -- plus a free-text QA note for mixed-bag/contaminated topics.
    is_excluded          BOOLEAN NOT NULL DEFAULT FALSE,
    exclusion_reason       VARCHAR(200) NULL,
    theme_group              VARCHAR(100) NULL,
    polarity_type              ENUM('defect','preference_split','conditional_fit') NULL,
    qa_note                      TEXT NULL
)
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;


-- ============================================================
-- 6. product_complaint_theme_mentions_v1 — mart table #3b (physical)
--    Complaint-theme table. Grain: topic_id x
--    product_id x review_month. avg_segment_sentiment reuses
--    cons_net_sentiment already computed in review_sentiment_scores_v1
--    (no re-scoring).
--
--    strong_complaint_share reframes avg_segment_sentiment -- which is
--    effectively trimodal (confident strong_complaint ~= -1.0, confident
--    minor_complaint ~= -0.5, confident usage_tip/positive ~= 0), not
--    continuous -- as a 0-1 severity share instead. See the column's own
--    comment below.
-- ============================================================

CREATE TABLE IF NOT EXISTS product_complaint_theme_mentions_v1 (
    topic_id                  INT NOT NULL,
    product_id                 BIGINT NOT NULL,
    review_month                 DATE NOT NULL,

    mention_count                  INT NOT NULL,  -- COUNT(DISTINCT review_id) per (topic_id, product_id, review_month),
                                                    -- NOT a clause-row count: sentence-splitting has no
                                                    -- concept of topic continuity, so one review can produce multiple
                                                    -- same-topic clause-documents (elaboration across sentences); counting
                                                    -- clause-rows would double/triple-count a single reviewer's one
                                                    -- complaint relative to a terse one-sentence version of the same
                                                    -- complaint.
    avg_segment_sentiment            DECIMAL(6,5) NOT NULL,
    representative_phrase              TEXT NULL,   -- BERTopic representative-document excerpt for this cell

    -- A 0-1 rescaling of avg_segment_sentiment's real signal:
    -- avg_segment_sentiment ~= -0.5 - 0.5*(strong_complaint_share) -- i.e. the
    -- -0.50..-0.60 band a naive chart would show IS the 0%-100% severity range,
    -- just compressed into the second decimal place. GENERATED ALWAYS AS (...)
    -- STORED, same self-populating pattern object #1 uses for cons_net_sentiment
    -- etc. -- no load-script logic, no drift risk. Floors to 0 for every cell at
    -- or above -0.5 avg_segment_sentiment -- ~76% of cells
    -- (20,209/26,624) -- because -0.5 is
    -- segment_net_sentiment's own "all confident minor_complaint" fixed point
    -- (0/-0.5/-1.0 for tip/minor/strong by construction), not an arbitrary
    -- cutoff: it correctly catches both the confident-minor_complaint mode and
    -- the small non-complaint-leakage tail near 0, not just the latter.
    strong_complaint_share            DECIMAL(5,4)
        GENERATED ALWAYS AS (
            GREATEST(0, (-avg_segment_sentiment - 0.5) / 0.5)
        ) STORED,

    PRIMARY KEY (topic_id, product_id, review_month),
    INDEX idx_product_month (product_id, review_month),

    CONSTRAINT fk_theme_mentions_topic_v1
        FOREIGN KEY (topic_id)
        REFERENCES complaint_topics_v1(topic_id)
        ON DELETE CASCADE,
    CONSTRAINT fk_theme_mentions_product_v1
        FOREIGN KEY (product_id)
        REFERENCES products_v3(product_id)
)
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;


-- ============================================================
-- 7. Sanity checks — run after loading real data and compare
--    against expectations, same pattern as mysql_preprocessing.sql.
-- ============================================================

-- SELECT COUNT(*) FROM review_sentiment_scores_v1;                 -- expect 814,443 (NOT 814,452 -- 9 reviews with no Korean text anywhere are a deliberate exclusion from scoring)
-- SELECT COUNT(*) FROM product_monthly_sentiment_v1;                -- expect a few thousand (103 products x ~months each)
-- SELECT * FROM voc_decline_risk_v1 WHERE voc_decline_flag = TRUE ORDER BY voc_decline_risk_score ASC LIMIT 20;
-- SELECT * FROM product_confidence_flags_v1 WHERE low_confidence_product = TRUE;
-- SELECT product_id, COUNT(*) AS category_products
--     FROM products_v3 GROUP BY category_id HAVING category_products >= 3;  -- category comparison gate, query-time only
--
-- These are quick starting points, not a full QA pass.
