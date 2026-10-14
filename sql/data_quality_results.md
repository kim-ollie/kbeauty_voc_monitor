# Data overview: results

Output of [`data_overview.sql`](data_overview.sql).
## Reviews and products

| Reviews | Products | Categories | Brands | Products without category |
|---|---|---|---|---|
| 814,452 | 103 | 26 | 55 | 0 |

| Reviews without date | First review | Last review |
|---|---|---|
| 0 | May 2014 | 2026-07-06 (July 2026 is a partial month; June 2026 is the anchor) |

## Ratings

| Rating | Reviews | % |
|---|---|---|
| 1 | 7,497 | 0.92 |
| 2 | 9,753 | 1.20 |
| 3 | 44,768 | 5.50 |
| 4 | 165,299 | 20.30 |
| 5 | 587,135 | 72.09 |

## Categories and product history

9 categories have 3 or more products. The 5 largest categories hold 66 of the 103 products.

How long each product had been reviewed by June 2026 (the decline monitor needs 4+ months of history):

| History | Products |
|---|---|
| Under 4 months | 7 |
| 4–11 months | 13 |
| 12–23 months | 17 |
| 24+ months | 66 |

## Text the model reads

Korean pros, cons and usage tip, after preprocessing.

| Reviews | Without Korean text | Pros | Cons | Tips | All segments | Left with double spaces | Left with 3+ line breaks |
|---|---|---|---|---|---|---|---|
| 814,452 | 9 | 814,443 | 814,443 | 238,292 | 1,867,178 | 0 | 0 |

Segment length in characters:

| Segment | Segments | p50 | p90 | p99 | Max |
|---|---|---|---|---|---|
| Pros | 814,443 | 54 | 292 | 746 | 5,652 |
| Cons | 814,443 | 39 | 93 | 251 | 2,986 |
| Tip | 238,292 | 45 | 120 | 299 | 6,099 |

## Model scores

| Class | Reviews | % |
|---|---|---|
| Positive | 696,856 | 85.6 |
| Neutral | 100,015 | 12.3 |
| Negative | 17,572 | 2.2 |

## Complaint themes, July 2025 – June 2026

Uncategorized mentions are excluded.

| Theme | Mentions | % of themed mentions | % strong |
|---|---|---|---|
| Insufficient moisturization | 22,165 | 18.1 | 1.6 |
| Texture & finish | 17,429 | 14.3 | 3.4 |
| No visible improvement | 12,719 | 10.4 | 6.0 |
| Packaging & dispensing | 10,564 | 8.6 | 1.1 |
| Adverse skin reaction | 9,862 | 8.1 | 32.8 |
| Capacity: too small, wants larger SKU | 9,781 | 8.0 | 0.3 |
| Price: too expensive | 7,878 | 6.4 | 0.9 |
| Cleansing performance | 7,781 | 6.4 | 1.2 |
| Too heavy / oily-skin mismatch | 6,413 | 5.2 | 3.3 |
| Scent | 6,227 | 5.1 | 1.5 |
| Price: discount-dependent purchase | 5,145 | 4.2 | 0.1 |
| Skin-type & seasonal fit | 3,368 | 2.8 | 1.3 |
| Application effort | 2,839 | 2.3 | 1.3 |
