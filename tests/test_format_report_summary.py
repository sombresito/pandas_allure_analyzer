import re
from report_summary import format_report_summary


def test_total_in_summary_blue():
    case = {"labels": [], "status": "passed", "time": {"start": 1}}
    summary = format_report_summary([case], color=True, fallback_timestamp=0)
    assert re.search(r'<span style="color:blue;">total=1</span>', summary)
