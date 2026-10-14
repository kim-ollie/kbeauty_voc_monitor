-- ============================================================
-- Hwahae review text preprocessing -- pure MySQL, no Python
-- ============================================================
--
-- Same cleaning contract as text_preprocessing.py, reimplemented as a
-- MySQL 8 stored FUNCTION + VIEW so it runs inside the database as its
-- own pipeline stage:
--
--   Hwahae reviews (collected)
--     -> MySQL normalized layer          (hwahae_reviews_raw_v3)
--     -> MySQL PREPROCESSING layer  <---  this file
--     -> MySQL analytics mart layer
--     -> RoBERTa / KcELECTRA sentiment classification
--     -> ...
--
-- Requirements:
--   - MySQL 8.0.4+ (REGEXP_REPLACE / REGEXP_LIKE use the ICU regex
--     engine only from 8.0.4 onward -- earlier 8.0.x and all 5.7 use
--     the old, byte-oriented regex library and this will not behave
--     correctly on Korean text). Tested against 8.0.46.
--   - Run `SET NAMES utf8mb4 COLLATE utf8mb4_unicode_ci;` at the start
--     of every session that uses these objects. Without it you get
--     "Illegal mix of collations (utf8mb4_unicode_ci,COERCIBLE) and
--     (utf8mb4_0900_ai_ci,COERCIBLE)" -- MySQL 8's *session* default
--     collation (utf8mb4_0900_ai_ci) does not match this project's
--     *schema* collation (utf8mb4_unicode_ci, set in hwahae_schema.sql),
--     and REGEXP comparisons refuse to mix them. This is not
--     hypothetical: the function below returned NULL for every input
--     the first time it was run against the real table, purely
--     because of this mismatch.
--
-- Same preservation rule as the Python version:
-- do NOT strip emoticons (ㅋㅋ, ㅠㅠ), elongated characters (좋아아아),
-- or informal punctuation (;; ...) -- they carry sentiment signal.
-- Nothing here touches negation, emoji, emoticons, elongation, tone
-- punctuation, or ingredient/product names -- only whitespace/line-
-- break/invisible-character noise.
--
-- One real gap versus the Python version: MySQL has no built-in
-- Unicode NFC normalization function at all (no equivalent of Python's
-- unicodedata.normalize('NFC', ...)). The 12K-row sample was checked
-- and was already 100% NFC at scrape time, so this is a documented,
-- accepted gap for this dataset -- not a silent omission. If the full
-- 814K-row table ever shows NFD-decomposed Hangul (combining jamo),
-- that would need a Python pass; there is no pure-SQL fix for it.
--
-- All counts in the comments below were produced by actually running
-- this file against a MySQL 8.0.46 instance loaded with the real
-- 12,000-row sample -- not estimated by hand.
-- ============================================================

SET NAMES utf8mb4 COLLATE utf8mb4_unicode_ci;


-- ------------------------------------------------------------
-- 1. clean_review_text(): whitespace / line-break / invisible-
--    character normalization. Apply to pros_en, cons_en, tip_en,
--    pros_ko, cons_ko, tip_ko.
-- ------------------------------------------------------------

DROP FUNCTION IF EXISTS clean_review_text;

DELIMITER $$

CREATE FUNCTION clean_review_text(input_text TEXT)
RETURNS TEXT
CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
DETERMINISTIC
BEGIN
    DECLARE result TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

    IF input_text IS NULL OR TRIM(input_text) = '' THEN
        RETURN NULL;
    END IF;

    SET result = input_text;

    -- Invisible / non-breaking characters -> normal space or dropped.
    -- NOTE: CHAR(N USING utf8mb4) does NOT encode code point N into
    -- utf8mb4 the way it looks like it should -- it treats N as a raw
    -- byte and tries to *reinterpret* that byte as UTF-8 input, which
    -- is invalid for any N > 127 and silently returns NULL under
    -- strict SQL mode. This is a real footgun, confirmed by testing
    -- (clean_review_text('hello world') returned NULL the first time,
    -- for exactly this reason), not a hypothetical warning. The
    -- correct way to build a specific Unicode character is
    -- CONVERT(UNHEX(utf8_hex_bytes) USING utf8mb4).
    SET result = REPLACE(result, CONVERT(UNHEX('C2A0') USING utf8mb4), ' ');    -- U+00A0 NBSP
    SET result = REPLACE(result, CONVERT(UNHEX('E2808B') USING utf8mb4), '');   -- U+200B zero-width space
    SET result = REPLACE(result, CONVERT(UNHEX('E2808C') USING utf8mb4), '');   -- U+200C ZWNJ
    SET result = REPLACE(result, CONVERT(UNHEX('E2808D') USING utf8mb4), '');   -- U+200D ZWJ
    SET result = REPLACE(result, CONVERT(UNHEX('EFBBBF') USING utf8mb4), '');   -- U+FEFF BOM / zero-width NBSP
    SET result = REPLACE(result, CHAR(9), ' ');                                 -- tab -> space (plain ASCII, safe as-is)

    -- Line breaks: CRLF / lone CR -> LF
    SET result = REPLACE(result, CONCAT(CHAR(13), CHAR(10)), CHAR(10));
    SET result = REPLACE(result, CHAR(13), CHAR(10));

    -- Collapse duplicated horizontal whitespace (tabs already -> spaces above)
    SET result = REGEXP_REPLACE(result, ' {2,}', ' ');

    -- Trim space at the edges of every line. Written as four one-
    -- directional passes instead of a single backreference pattern
    -- (e.g. '(^|\n)[ ]+' -> '\1') -- MySQL's ICU REGEXP_REPLACE
    -- backreference syntax in the replacement string is easy to get
    -- subtly wrong, and these four are simpler to read and verify.
    SET result = REGEXP_REPLACE(result, '\n[ ]+', '\n');
    SET result = REGEXP_REPLACE(result, '[ ]+\n', '\n');
    SET result = REGEXP_REPLACE(result, '^[ ]+', '');
    SET result = REGEXP_REPLACE(result, '[ ]+$', '');

    -- Collapse runs of 3+ newlines down to a single blank line (\n\n).
    -- Genuine \n\n (reviewer-inserted paragraph breaks, e.g. under a
    -- self-written "<Hydration>" header) is preserved on purpose.
    SET result = REGEXP_REPLACE(result, '\n{3,}', '\n\n');
    -- Re-run once more: trimming line-edge spaces above can unmask a
    -- "\n<space>\n<space>\n" run (not caught by the first \n{3,} pass,
    -- since the spaces break up the run) into a bare \n\n\n run. Found
    -- this by testing against real review text, not by inspection --
    -- 4 rows in the 12K sample hit this exact case.
    SET result = REGEXP_REPLACE(result, '\n{3,}', '\n\n');

    SET result = TRIM(result);

    IF result = '' THEN
        RETURN NULL;
    END IF;

    RETURN result;
END$$

DELIMITER ;


-- ------------------------------------------------------------
-- 2. hwahae_reviews_clean_v3: non-destructive preprocessing layer.
--    hwahae_reviews_raw_v3 is untouched (keeps the scrape-time audit
--    trail intact, matching the project's existing raw-vs-view
--    pattern used by hwahae_reviews_dataset_v3). This view sits
--    alongside that view, not inside it -- join the two at query time
--    if you need product enrichment + cleaned text together.
--
--    Verified against the 12K-row sample: output is byte-for-byte
--    consistent with text_preprocessing.py's clean_review_text() --
--    same before/after counts for double-space runs (1,104 -> 0),
--    3+-newline runs (160 -> 0), lang_routing_mismatch (5 rows),
--    and cons_ko_empty_complaint_risk (2,716 rows).
-- ------------------------------------------------------------

CREATE OR REPLACE VIEW hwahae_reviews_clean_v3 AS
SELECT
    r.review_id,
    r.product_id,
    r.rating,

    clean_review_text(r.pros_en) AS pros_en,
    clean_review_text(r.cons_en) AS cons_en,
    clean_review_text(r.tip_en)  AS tip_en,
    CONCAT_WS('\n',
        clean_review_text(r.pros_en),
        clean_review_text(r.cons_en),
        clean_review_text(r.tip_en)
    ) AS review_text_en_combined,

    clean_review_text(r.pros_ko) AS pros_ko,
    clean_review_text(r.cons_ko) AS cons_ko,
    clean_review_text(r.tip_ko)  AS tip_ko,
    CONCAT_WS('\n',
        clean_review_text(r.pros_ko),
        clean_review_text(r.cons_ko),
        clean_review_text(r.tip_ko)
    ) AS review_text_ko_combined,

    r.has_korean_detail,
    r.is_translated,

    -- -------- validation flags: triage signals, NOT ground truth --------
    -- Ported from text_preprocessing.py's build_validation_report().
    -- Spot-checking in Python found a real false-positive rate on
    -- these (26.3% of cons_ko_empty_complaint_risk hits still contain
    -- a genuine hedged complaint, e.g. "입구쪽이 약간 불편한것 말곤
    -- 없었어요" = "nothing except the opening's a bit awkward"). Use
    -- these to route rows to manual/LLM review, same as the Python
    -- version -- do not auto-relabel or auto-exclude on them alone.

    -- is_translated = 0 is documented to mean "_en holds untranslated
    -- Korean, route via _ko" -- but 5/75 such rows in the sample have
    -- ENGLISH text in _ko too. Hangul fraction = (hangul chars) /
    -- (hangul + latin chars); flagged when is_translated = 0 but
    -- pros_ko has zero Hangul and at least one Latin letter.
    (r.is_translated = 0
     AND (CHAR_LENGTH(r.pros_ko) - CHAR_LENGTH(REGEXP_REPLACE(r.pros_ko, '[가-힣]', ''))) = 0
     AND (CHAR_LENGTH(r.pros_ko) - CHAR_LENGTH(REGEXP_REPLACE(r.pros_ko, '[A-Za-z]', ''))) > 0
    ) AS lang_routing_mismatch,

    (r.cons_ko REGEXP '없어요|없습니다|없네요|딱히 ?없|특별히 ?없') AS cons_ko_empty_complaint_risk,
    (r.cons_en REGEXP 'none|nothing to|no downsides?|no complaints?|no cons\\b')
        AS cons_en_empty_complaint_risk,

    -- Reviews still pending Hwahae's own moderation, or hidden from
    -- public display -- text is present but not platform-finalized.
    CASE
        WHEN r.is_validated = 0 THEN 'pending_validation'
        WHEN r.blind = 1        THEN 'blind_hidden'
        ELSE NULL
    END AS platform_status

FROM hwahae_reviews_raw_v3 r;


-- ------------------------------------------------------------
-- 3. Sanity checks -- run these after creating the view on the real
--    table and compare against the numbers in the comments above.
-- ------------------------------------------------------------

-- SELECT COUNT(*) AS total_rows FROM hwahae_reviews_clean_v3;
--
-- SELECT
--   SUM(pros_en REGEXP '  +') + SUM(cons_en REGEXP '  +') + SUM(tip_en REGEXP '  +') AS before_double_space
-- FROM hwahae_reviews_raw_v3;
--
-- SELECT
--   SUM(pros_en REGEXP '  +') + SUM(cons_en REGEXP '  +') + SUM(tip_en REGEXP '  +') AS after_double_space
-- FROM hwahae_reviews_clean_v3;
--
-- SELECT SUM(lang_routing_mismatch), SUM(cons_ko_empty_complaint_risk),
--        SUM(platform_status IS NOT NULL)
-- FROM hwahae_reviews_clean_v3;


-- ------------------------------------------------------------
-- 4. Performance note (measured, not guessed)
-- ------------------------------------------------------------
-- A full scan of this view over the 12,000-row sample -- COUNT(*) +
-- SUM(CHAR_LENGTH(review_text_ko_combined)), forcing every row's
-- REGEXP chain to actually run -- took ~2.5s wall clock including
-- MySQL client connection overhead. That's several REGEXP_REPLACE
-- calls per row across 6 text columns, recomputed on every query
-- since this is a plain (non-materialized) VIEW.
--
-- At 814,452 rows (~68x this sample) a naive extrapolation puts a
-- full scan somewhere in the tens of seconds, not milliseconds. Fine
-- for an occasional batch pull; likely too slow if the dashboard or
-- mart-building queries hit this view repeatedly. If that turns out
-- to matter in practice:
--   CREATE TABLE hwahae_reviews_clean_v3_materialized AS
--   SELECT * FROM hwahae_reviews_clean_v3;
-- and refresh it on a schedule (matching the existing project's
-- pattern of scheduled/batch refreshes rather than live queries)
-- instead of querying the view directly.
