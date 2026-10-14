"""The line under a flagged product's number row: how long it has been flagged, and since when its score has been running.

Pure functions over one watchlist entry of the saved payloads: no model call, no database, no network.

    Flagged 3 months in a row · below its recent average since Jan 2025
    First flagged this month · below its recent average since Feb 2026

The first half is the payload's streak of consecutive months at tier 1 or 2. The second half is `decline_since`: the first month of
the current unbroken run of a positive CUSUM score. The line says what the score measures, a shortfall against the product's own
trailing 6-month average, rather than "declining since": the score can start with a small dip months before the real drop. It is a
plain-language paraphrase: the score can stay positive in a month that is slightly above the average, so "below its recent average
since" is not a claim that every month since was below it.

Every number printed is a string copied from the payload; the month is the payload's own month name cut to three letters.
"""
TAIL = "below its recent average since"
FIRST = "First flagged this month"


def short_month(display):
    """'January 2025' -> 'Jan 2025'."""
    month, year = display.split()
    return f"{month[:3]} {year}"


def flag_line(entry):
    """The line for a tier 1-2 entry. A tier-3-only entry has no streak and gets none (ValueError)."""
    n = entry["consecutive_months_flagged_tier_1_2"]
    if n is None:
        raise ValueError("a tier-3-only product has no flag line")
    head = FIRST if n == 1 else f"Flagged {entry['consecutive_months_flagged_tier_1_2_display']} months in a row"
    since = entry["decline_since_display"]
    return head if not since else f"{head} · {TAIL} {short_month(since)}"


def verify_flag_line(line, entry):
    """Problems with a line against its entry; [] means all clear."""
    problems = []
    if entry["highest_tier"] not in (1, 2):
        return ["the entry is not tier 1 or 2"]
    n = entry["consecutive_months_flagged_tier_1_2"]
    if n == 1 and not line.startswith(FIRST):
        problems.append("a first-month product must say so")
    if n != 1 and not line.startswith(f"Flagged {entry['consecutive_months_flagged_tier_1_2_display']} months in a row"):
        problems.append("the streak is not the payload's")
    since = entry["decline_since_display"]
    if since and not line.endswith(f"{TAIL} {short_month(since)}"):
        problems.append("the month is not the payload's decline_since")
    if not since and TAIL in line:
        problems.append("a since-month was printed although the payload has none")
    return problems


def word_count(line):
    return len(line.split())
