-- Data overview. Results: data_quality_results.md
SET NAMES utf8mb4 COLLATE utf8mb4_unicode_ci;


-- Reviews and products
SELECT (SELECT COUNT(*) FROM hwahae_reviews_raw_v3) AS reviews,
       COUNT(*)                    AS products,
       COUNT(DISTINCT category_id) AS categories,
       COUNT(DISTINCT brand_id)    AS brands,
       SUM(category_id IS NULL)    AS products_without_category
FROM products_v3;

SELECT SUM(review_date IS NULL)                  AS reviews_without_date,
       DATE_FORMAT(MIN(review_date), '%Y-%m')    AS first_review,
       DATE_FORMAT(MAX(review_date), '%Y-%m-%d') AS last_review
FROM hwahae_reviews_raw_v3;


-- Ratings
SELECT SUM(rating IS NULL OR rating NOT BETWEEN 1 AND 5) AS invalid_ratings,
       ROUND(100 * SUM(rating >= 4) / COUNT(*), 2)       AS pct_4_or_5_stars
FROM hwahae_reviews_raw_v3;

SELECT rating, COUNT(*) AS reviews,
       ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM hwahae_reviews_raw_v3
GROUP BY rating
ORDER BY rating;


-- Categories and product history
SELECT SUM(n >= 3) AS categories_with_3plus_products,
       SUM(CASE WHEN rnk <= 5 THEN n ELSE 0 END) AS products_in_5_largest
FROM (SELECT COUNT(*) AS n, ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC) AS rnk
      FROM products_v3
      GROUP BY category_id) c;

-- Months reviewed by June 2026
SELECT CASE WHEN m < 4  THEN 'under 4 months'
            WHEN m < 12 THEN '4-11 months'
            WHEN m < 24 THEN '12-23 months'
            ELSE             '24+ months' END AS history,
       COUNT(*) AS products
FROM (SELECT TIMESTAMPDIFF(MONTH, DATE_FORMAT(MIN(review_date), '%Y-%m-01'), '2026-06-01') + 1 AS m
      FROM hwahae_reviews_raw_v3
      GROUP BY product_id) p
GROUP BY history
ORDER BY MIN(m);


-- Text the model reads (Korean pros / cons / tip after preprocessing)
SELECT COUNT(*) AS reviews,
       SUM(pros_ko IS NULL AND cons_ko IS NULL AND tip_ko IS NULL) AS without_korean_text,
       COUNT(pros_ko) AS pros,
       COUNT(cons_ko) AS cons,
       COUNT(tip_ko)  AS tips,
       COUNT(pros_ko) + COUNT(cons_ko) + COUNT(tip_ko) AS all_segments,
       SUM(CONCAT_WS(' ', pros_ko, cons_ko, tip_ko) REGEXP '  ')       AS left_with_double_spaces,
       SUM(CONCAT_WS(' ', pros_ko, cons_ko, tip_ko) REGEXP '\\n{3,}')  AS left_with_3plus_line_breaks
FROM hwahae_reviews_clean_v3;

-- Segment length in characters
WITH v AS (
    SELECT CHAR_LENGTH(pros_ko) AS pros, CHAR_LENGTH(cons_ko) AS cons, CHAR_LENGTH(tip_ko) AS tip
    FROM hwahae_reviews_clean_v3
),
lengths AS (
    SELECT /*+ NO_MERGE(v) */ s.segment,
           CASE s.segment WHEN 'pros' THEN v.pros WHEN 'cons' THEN v.cons ELSE v.tip END AS len
    FROM v
    CROSS JOIN (SELECT 'pros' AS segment UNION ALL SELECT 'cons' UNION ALL SELECT 'tip') s
),
ranked AS (
    SELECT segment, len, PERCENT_RANK() OVER (PARTITION BY segment ORDER BY len) AS pct
    FROM lengths
    WHERE len IS NOT NULL
)
SELECT segment, COUNT(*) AS segments,
       MIN(CASE WHEN pct >= 0.50 THEN len END) AS p50,
       MIN(CASE WHEN pct >= 0.90 THEN len END) AS p90,
       MIN(CASE WHEN pct >= 0.99 THEN len END) AS p99,
       MAX(len) AS max_len
FROM ranked
GROUP BY segment
ORDER BY FIELD(segment, 'pros', 'cons', 'tip');


-- Model scores: predicted class, review-level model
SELECT predicted_class, COUNT(*) AS reviews,
       ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
FROM (SELECT CASE
               WHEN review_p_positive >= review_p_neutral AND review_p_positive >= review_p_negative THEN 'positive'
               WHEN review_p_negative >= review_p_neutral AND review_p_negative >= review_p_positive THEN 'negative'
               ELSE 'neutral'
             END AS predicted_class
      FROM review_sentiment_scores_v1) c
GROUP BY predicted_class
ORDER BY FIELD(predicted_class, 'positive', 'neutral', 'negative');


-- Complaint themes, July 2025 - June 2026 (uncategorized excluded)
SELECT t.theme_group,
       SUM(x.mention_count) AS mentions,
       ROUND(100 * SUM(x.mention_count) / SUM(SUM(x.mention_count)) OVER (), 1) AS pct_of_themed_mentions,
       ROUND(100 * SUM(x.mention_count * x.strong_complaint_share) / SUM(x.mention_count), 1) AS pct_strong
FROM product_complaint_theme_mentions_v1 x
JOIN complaint_topics_v1 t ON t.topic_id = x.topic_id
WHERE x.topic_id <> -1
GROUP BY t.theme_group
ORDER BY mentions DESC;
