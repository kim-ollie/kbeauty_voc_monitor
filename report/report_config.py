"""Settings the report reads from outside the payloads.

THE DASHBOARD LINK. `DASHBOARD_URL` is the public Tableau dashboard. When it is empty the report has no link
line at all. When a URL is set, the message gets one small line under
its header (a Slack `context` block holding `<url|label>`; in the Markdown file `[label](url)`), and gates.py checks that the line is there once, with
exactly this URL and this label, and that no other link is anywhere in the payload.

The URL is not a secret (a public dashboard) and it is not data, so it lives here and not in a payload: the builder and the twelve payloads stay as they are.
A Slack webhook cannot edit a post, so a message already posted keeps whatever line it had; only months rendered and posted after the URL is set carry it.
The link opens the dashboard as it stands (data to June 2026), also from a report for an earlier month.

What is accepted (anything else raises ConfigProblem at once, so a typo cannot reach a report):
    URL    https:// then a host, then an optional path (a `?` and simple parameters are allowed); no spaces and none of  < > | & ( ) " ' \\ `
           (so no `&`: leave the extra parameters off a Tableau share link, everything from the first `&` on is not needed)
    label  1 to 60 characters, one line, none of  < > | & * _ ~ ` (they are Slack markup)
"""
import re

DASHBOARD_URL = "https://public.tableau.com/views/ProjectDashboard_17905074206370/Primary?:language=en-US"
DASHBOARD_LABEL = "Dashboard"

URL_PATTERN = re.compile(r"^https://[A-Za-z0-9][A-Za-z0-9.-]*(:\d+)?(/[A-Za-z0-9\-._~%/:@+=,;!$*#?\[\]]*)?\Z")
LABEL_FORBIDDEN = set("<>|&*_~`")
LABEL_MAX = 60


class ConfigProblem(ValueError):
    """The dashboard setting is not usable. The message names the setting, never a customer's or anyone's words."""


def check_url(url):
    if not isinstance(url, str) or not URL_PATTERN.match(url):
        raise ConfigProblem("DASHBOARD_URL must be an https:// address with no spaces and none of < > | & ( ) quotes or backticks; leave the query parameters off")
    if len(url) > 500:
        raise ConfigProblem("DASHBOARD_URL is longer than 500 characters")
    return url


def check_label(label):
    if not isinstance(label, str) or not 1 <= len(label) <= LABEL_MAX or "\n" in label or "\r" in label or label != label.strip():
        raise ConfigProblem(f"DASHBOARD_LABEL must be one line of 1 to {LABEL_MAX} characters with no spaces at either end")
    bad = sorted(set(label) & LABEL_FORBIDDEN)
    if bad:
        raise ConfigProblem(f"DASHBOARD_LABEL holds Slack markup characters {bad}")
    return label


def dashboard(url=None, label=None):
    """None while no URL is set; otherwise {"url", "label"}, both checked. `url` and `label` exist for tests; the report calls it with no arguments."""
    url = DASHBOARD_URL if url is None else url
    label = DASHBOARD_LABEL if label is None else label
    if not url:
        return None
    return {"url": check_url(url), "label": check_label(label)}
