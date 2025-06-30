import pandas as pd


def extract_test_cases(node):
    """Recursively extract all test case dictionaries from allure tree."""
    raw = []
    if isinstance(node, dict):
        if "uid" in node:
            raw.append(node)
        for child in node.get("children", []):
            raw.extend(extract_test_cases(child))
    elif isinstance(node, list):
        for item in node:
            raw.extend(extract_test_cases(item))
    return raw


def chunk_json_to_jsonl(data, output_path: str, report_uuid: str) -> pd.DataFrame:
    """Convert allure JSON data to a JSONL file of test case chunks.

    Parameters
    ----------
    data : Any
        Parsed JSON structure with allure report data.
    output_path : str
        Destination path for resulting JSONL file.
    report_uuid : str
        Identifier of the report to be added to each row.
    """
    raw = extract_test_cases(data)

    df = pd.DataFrame([
        {
            "uid": t.get("uid"),
            "name": t.get("name"),
            "parentSuite": next(
                (l["value"] for l in t.get("labels", []) if l.get("name") == "parentSuite"),
                "unknown",
            ),
            "suite": next(
                (l["value"] for l in t.get("labels", []) if l.get("name") == "suite"),
                "unknown",
            ),
            "owner": next(
                (l["value"] for l in t.get("labels", []) if l.get("name") == "owner"),
                "unknown",
            ),
            "severity": next(
                (l["value"] for l in t.get("labels", []) if l.get("name") == "severity"),
                "unknown",
            ),
            "feature": next(
                (l["value"] for l in t.get("labels", []) if l.get("name") == "feature"),
                "unknown",
            ),
            "host": next(
                (l["value"] for l in t.get("labels", []) if l.get("name") == "host"),
                "unknown",
            ),
            "status": t.get("status"),
            "statusMessage": (t.get("statusMessage") or "").strip(),
            "statusTrace": (t.get("statusTrace") or "").strip(),
        }
        for t in raw
    ])

    df["rag_text"] = df.apply(
        lambda row: (
            f"Название теста: {row['name']}\n"
            f"Команда: {row['parentSuite']}\n"
            f"Модуль: {row['suite']}\n"
            f"Владелец: {row['owner']}\n"
            f"Серьёзность: {row['severity']}\n"
            f"Фича: {row['feature']}\n"
            f"Хост: {row['host']}\n"
            f"Статус: {row['status']}\n"
            f"Сообщение: {row['statusMessage']}\n"
            f"Трейс: {row['statusTrace']}"
        ),
        axis=1,
    )

    df["report_uuid"] = report_uuid

    df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    return df
