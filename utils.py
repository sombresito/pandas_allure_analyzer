import os
import logging
from fastapi import HTTPException
from pandas_chunking import chunk_json_to_jsonl
from rag_pipeline import run_rag_analysis, RagAnalysisError
from report_summary import format_report_summary
import requests
import unicodedata
import re
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

# Базовый URL Allure API, по умолчанию взят из переменной окружения
ALLURE_API = os.getenv("ALLURE_API", "https://Allure-Report-BCC-QA.bank.corp.centercredit.kz:9443/api")

def _auth_kwargs():
    """
    Возвращает kwargs для запросов к Allure: либо заголовок Bearer,
    либо basic-auth, либо пустой dict
    """
    token = os.getenv("ALLURE_TOKEN")
    if token:
        return {"headers": {"Authorization": f"Bearer {token}"}}

    user = os.getenv("ALLURE_USER")
    password = os.getenv("ALLURE_PASS")
    if user and password:
        return {"auth": (user, password)}

    return {}

def extract_test_suite_name(report_json):
    """
    Извлекает название команды из JSON-отчёта:
    сначала пытается взять метку parentSuite,
    если её нет — берёт метку suite.
    Очищает результат, оставляя цифры, латиницу,
    кириллицу, пробелы и знаки ':' и '-'.
    """
    def _search_label(node, label_name):
        if isinstance(node, dict):
            for label in node.get("labels", []):
                if label.get("name") == label_name:
                    return label.get("value")
            for value in node.values():
                if isinstance(value, (dict, list)):
                    found = _search_label(value, label_name)
                    if found:
                        return found
        elif isinstance(node, list):
            for item in node:
                found = _search_label(item, label_name)
                if found:
                    return found
        return None

    # Сначала пытаемся взять parentSuite, иначе — suite
    raw = _search_label(report_json, "parentSuite") or _search_label(report_json, "suite")
    if not raw:
        return None

    # Нормализуем Unicode
    name = unicodedata.normalize("NFC", raw)
    # Очищаем строку
    name = re.sub(r"[^0-9A-Za-zА-Яа-яЁё\s:\-]", "", name)
    return name


def chunk_and_save_json(json_data, uuid, test_suite_name):
    """
    Разбивает JSON-отчёт на чанки, сохраняет в <chunks>/<test_suite_name>/<uuid>.jsonl
    и удаляет старые файлы, оставляя не более 3.
    """
    base_dir = os.path.join("chunks", test_suite_name)
    os.makedirs(base_dir, exist_ok=True)

    output_path = os.path.join(base_dir, f"{uuid}.jsonl")
    df = chunk_json_to_jsonl(json_data, output_path, uuid)

    # Оставляем только последние 3 отчёта
    files = [os.path.join(base_dir, f) for f in os.listdir(base_dir)]
    files.sort(key=os.path.getmtime, reverse=True)
    while len(files) > 3:
        os.remove(files.pop())

    return output_path, df


def analyze_and_post(
    uuid: str,
    test_suite_name: str,
    report_data,
    question_override: str | None = None,
    prompt_override: str | None = None,
):
    """
    Выполняет RAG-анализ и отправляет результат на Allure-сервер.

    Вход: POST /uuid/analyze с JSON {"uuid": "..."}
    GET Allure-отчёта: GET {ALLURE_API}/report/{uuid}/test-cases/aggregate
    Выход: POST {ALLURE_API}/analysis/report/{uuid} с JSON вида:
      [{"rule": "auto-analysis", "message": "<текст анализа>"}]
    """
    # 1. Генерируем анализ
    try:
        analysis_result = run_rag_analysis(
            test_suite_name,
            question_override,
            prompt_override,
        )
        analysis_text = analysis_result.get("analysis", "")
    except RagAnalysisError as e:
        logger.error("RAG analysis failed for %s: %s", uuid, e)
        raise HTTPException(
            status_code=500,
            detail=f"Qdrant service is unreachable: {e}"
        ) from e

    summary_text = format_report_summary(report_data)
    combined_text = summary_text + "\n\n" + analysis_text

    # Формируем полезную нагрузку по спецификации
    payload = [
        {
            "rule": "auto-analysis",
            "message": combined_text
        }
    ]

    # 2. Отправляем анализ на Allure-сервер
    url = f"{ALLURE_API}/analysis/report/{uuid}"
    auth_kwargs = _auth_kwargs()
    try:
        resp = requests.post(url, json=payload, verify=False, timeout=10, **auth_kwargs)
    except requests.RequestException as e:
        logger.error("Failed to post analysis for %s: %s", uuid, e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to post analysis: {e}"
        ) from e

    if not 200 <= resp.status_code < 300:
        msg = f"Unexpected status {resp.status_code}: {resp.text}"
        logger.error("Failed to post analysis for %s: %s", uuid, msg)
        raise HTTPException(status_code=500, detail=msg)

    logger.info("Analysis posted successfully for %s", uuid)
