"""Utilities to summarise a single Allure report."""

from collections import Counter
from typing import List, Dict, Any, Optional
from datetime import datetime

STATUS_ORDER = ["passed", "failed", "broken", "skipped"]
HTML_COLORS = {
    "passed": "green",
    "failed": "red",
    "broken": "orange",
    "skipped": "gray",
    "total": "blue",
}


def _format_date(ts: int) -> str:
    if ts <= 0:
        ts = 0
    return datetime.fromtimestamp(ts).strftime("%d.%m.%Y (%H:%M)")


def _normalize_timestamp(ts: float) -> int:
    if ts > 1e10:
        ts /= 1000.0
    return int(ts)


def extract_report_info(report: List[Dict[str, Any]], fallback_timestamp: int = 0) -> Dict[str, Any]:
    earliest = None
    test_suite_names = set()
    status_counts = Counter()
    initiators = set()
    jira_links = set()
    name_counter = Counter()

    for case in report:
        # timestamps
        t = case.get("time") or {}
        start = t.get("start")
        ts_fallback = case.get("timestamp")
        if isinstance(start, (int, float)):
            earliest = min(earliest or start, start)
        elif isinstance(ts_fallback, (int, float)):
            earliest = min(earliest or ts_fallback, ts_fallback)

        # labels → team & initiators
        suite_name = None
        fallback_suite = None
        for lbl in case.get("labels", []):
            name, val = lbl.get("name"), lbl.get("value")
            if name == "parentSuite" and val and suite_name is None:
                suite_name = val
            if name == "suite" and val and fallback_suite is None:
                fallback_suite = val
            if name in {"owner", "user", "initiator"} and val:
                initiators.add(val)
        if suite_name is None:
            suite_name = fallback_suite
        if suite_name:
            test_suite_names.add(suite_name)

        # status counts
        status = (case.get("status") or "").lower()
        if status in STATUS_ORDER:
            status_counts[status] += 1

        # jira links in links or jira field
        for link in case.get("links", []):
            if isinstance(link, dict):
                type_name = str(link.get("type") or link.get("name") or "").lower()
                if "jira" in type_name and link.get("url"):
                    jira_links.add(link["url"])
        jira_field = case.get("jira")
        if isinstance(jira_field, str):
            jira_links.add(jira_field)
        elif isinstance(jira_field, list):
            for j in jira_field:
                if isinstance(j, str):
                    jira_links.add(j)
                elif isinstance(j, dict):
                    url = j.get("url") or j.get("id") or j.get("name")
                    if url:
                        jira_links.add(str(url))

        # duplicate names
        if case.get("name"):
            name_counter[case["name"]] += 1

    duplicates = [n for n, c in name_counter.items() if c > 1]
    earliest = earliest or fallback_timestamp
    timestamp = _normalize_timestamp(earliest)

    # pick a single team name (or join them)
    if len(test_suite_names) == 1:
        test_suite_name = next(iter(test_suite_names))
    elif test_suite_names:
        test_suite_name = "_".join(sorted(test_suite_names))
    else:
        test_suite_name = ""

    return {
        "timestamp": timestamp,
        "test_suite_name": test_suite_name,
        "status_counts": {s: status_counts.get(s, 0) for s in STATUS_ORDER},
        "initiators": sorted(initiators),
        "jira_links": sorted(jira_links),
        "duplicates": sorted(duplicates),
    }


def _fmt_status(s: str, cnt: int, color: bool) -> str:
    if color and s in HTML_COLORS:
        return f'<span style="color:{HTML_COLORS[s]};">{s}={cnt}</span>'
    return f"{s}={cnt}"


def format_report_summary(
    report: List[Dict[str, Any]],
    color: bool = True,
    fallback_timestamp: Optional[int] = None,
) -> str:
    """Return a human-readable summary for a single Allure report.

    The summary starts with the report date, followed by one line per
    status count and optional metadata such as team name, initiators and
    jira links.
    """
    info = extract_report_info(report, fallback_timestamp or 0)

    date_str = _format_date(info["timestamp"])
    sc = info["status_counts"]
    # generate one line per status after the date line
    status_lines = [_fmt_status(s, sc[s], color) for s in STATUS_ORDER]
    total_tests = sum(sc.values())
    status_lines.append(_fmt_status("total", total_tests, color))

    lines = [f"**{date_str}**:"]
    lines.extend(status_lines)

    if info["test_suite_name"]:
        lines.append(f"**\u041a\u043e\u043c\u0430\u043d\u0434\u0430**: {info['test_suite_name']}")

    initiators = ", ".join(info["initiators"]) or "нет"
    lines.append(f"**\u0418\u043d\u0438\u0446\u0438\u0430\u0442\u043e\u0440\u044b**: {initiators}")

    for link in info["jira_links"]:
        lines.append(f"**jira**: {link}")

    if info["duplicates"]:
        dups = ", ".join(info["duplicates"])
        lines.append(f"**\u0414\u0443\u0431\u043b\u0438\u043a\u0430\u0442\u044b**: {dups}")
    else:
        lines.append("**\u0414\u0443\u0431\u043b\u0438\u043a\u0430\u0442\u044b**: нет")

    return "<br>".join(lines)
