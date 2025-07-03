from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import logging
# The main application orchestrates processing of Allure reports
from dotenv import load_dotenv
import urllib3
# Отключаем предупреждения InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# LOG_LEVEL is read before loading the .env file to match previous behaviour
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

load_dotenv()

from utils import (
    extract_test_suite_name,
    chunk_and_save_json,
    analyze_and_post,
    _auth_kwargs,
    ALLURE_API,
)
from embeddings import create_embeddings, load_chunks
from save_embeddings_to_qdrant import upload_embeddings
import rag_pipeline
app = FastAPI()


""" @app.post("/prompt")
async def set_prompt(request: Request):
    body = await request.json()
    prompt = body.get("prompt") or body.get("question")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt not provided.")
    rag_pipeline.question = str(prompt)
    logger.info("Analysis prompt updated")
    return {"Результат": "ok", "prompt": rag_pipeline.question} """

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],   
    allow_headers=["*"],   
    allow_credentials=True,
)

@app.post("/uuid/analyze")
async def analyze_report(request: Request):
    body = await request.json()
    uuid = body.get("uuid")
    if not uuid:
        raise HTTPException(status_code=400, detail="UUID not provided.")
    logger.info("Получен запрос на анализ для UUID %s", uuid)
    logger.info("Глобальный Промпт: %s", rag_pipeline.question)

    # 1. Получаем JSON отчёт
    url = f"{ALLURE_API}/report/{uuid}/test-cases/aggregate"
    auth_kwargs = _auth_kwargs()
    try:
        resp = requests.get(url, verify=False, timeout=10, **auth_kwargs)
        resp.raise_for_status()
        logger.info("Получены данные отчёта из Allure")
    except requests.RequestException as e:
        logger.error("Не удалось получить отчёт для %s: %s", uuid, e)
        raise HTTPException(status_code=500, detail=f"Не удалось получить отчёт: {e}") from e

    print(resp.status_code)
    print(repr(resp.text))
    
    try:
        report_data = resp.json()
    except ValueError as e:
        # сохраняем в файл для последующего анализа
        bad_path = f"/tmp/{uuid}_invalid_allure_response.txt"
        with open(bad_path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        logger.error("Получен некорректный JSON для %s, необработанный ответ сохранён в %s", uuid, bad_path)
        # пробрасываем понятную HTTP-ошибку
        raise HTTPException(
            status_code=502,
            detail=f"Получен некорректный JSON от Allure (см. {bad_path})"
        )

    # 2. Получаем название команды
    test_suite_name = extract_test_suite_name(report_data)
    if not test_suite_name:
        logger.error("Имя набора тестов не найдено в отчёте %s", uuid)
        raise HTTPException(status_code=400, detail="Team name (parentSuite) not found.")
    logger.info("Извлеченное имя набора тестов: %s", test_suite_name)

    # 3. Чанкуем и сохраняем
    json_path, df = chunk_and_save_json(report_data, uuid, test_suite_name)
    logger.info("Сохранены чанки для %s", uuid)

    # 4. Генерация и загрузка эмбеддингов
    try:
        if df is None:
            df = load_chunks(json_path)
        embeddings = create_embeddings(df)
        upload_embeddings(df, embeddings, test_suite_name, uuid)
        logger.info("Эмбеддинги загружены для %s", uuid)
    except Exception as e:
        logger.error("Не удалось загрузить эмбеддинги для %s: %s", uuid, e)
        raise HTTPException(status_code=500, detail=f"Не удалось загрузить эмбеддинги: {e}") from e

    # 5. Анализ и отправка результата
    try:
        analyze_and_post(uuid, test_suite_name, report_data)
        logger.info("Анализ отправлен для %s", uuid)
    except Exception as e:
        logger.error("Не удалось выполнить анализ для %s: %s", uuid, e)
        return {"Результат": "частичный", "Ошибка": f"{str(e)}\nСорян, не могу перевести ошибку на русский. Попробуй Google Translate, если языки — не твоё."}

    return {"Результат": "ok", "Набор тестов": test_suite_name}

@app.post("/prompt/analyze")
async def analyze_report_with_prompt(request: Request):
    """Analyze a report with a custom prompt applied only to this request."""
    body = await request.json()
    uuid = body.get("uuid")
    prompt = body.get("prompt") or body.get("question")
    if not uuid:
        raise HTTPException(status_code=400, detail="UUID не был предоставлен.")
    if not prompt:
        raise HTTPException(status_code=400, detail="Промпт не был предоставлен.")
    logger.info("Получен запрос анализа с промптом для UUID %s", uuid)
    logger.info("Отправленный промпт: %s", prompt)

    url = f"{ALLURE_API}/report/{uuid}/test-cases/aggregate"
    auth_kwargs = _auth_kwargs()
    try:
        resp = requests.get(url, verify=False, timeout=10, **auth_kwargs)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("Не удалось получить отчёт для %s: %s", uuid, e)
        raise HTTPException(status_code=500, detail=f"Не удалось получить отчёт: {e}") from e

    try:
        report_data = resp.json()
    except ValueError as e:
        bad_path = f"/tmp/{uuid}_invalid_allure_response.txt"
        with open(bad_path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        logger.error("Получен некорректный JSON для %s, необработанный ответ сохранён в %s", uuid, bad_path)
        raise HTTPException(status_code=502, detail=f"Получен некорректный JSON от Allure (см. {bad_path})")

    test_suite_name = extract_test_suite_name(report_data)
    if not test_suite_name:
        logger.error("Имя набора тестов не найдено в отчёте %s", uuid)
        raise HTTPException(status_code=400, detail="Имя набора тестов (parentSuite) не найдено.")

    json_path, df = chunk_and_save_json(report_data, uuid, test_suite_name)

    try:
        if df is None:
            df = load_chunks(json_path)
        embeddings = create_embeddings(df)
        upload_embeddings(df, embeddings, test_suite_name, uuid)
    except Exception as e:
        logger.error("Не удалось загрузить эмбеддинги для %s: %s", uuid, e)
        raise HTTPException(status_code=500, detail=f"Не удалось загрузить эмбеддинги: {e}") from e

    try:
        analyze_and_post(
            uuid,
            test_suite_name,
            report_data,
            prompt_override=prompt,
        )
    except Exception as e:
        logger.error("Не удалось выполнить анализ для %s: %s", uuid, e)
        return {"Результат": "частичный", "Ошибка": f"{str(e)}\nСорян, не могу перевести ошибку на русский. Попробуй Google Translate, если языки — не твоё."}

    return {"Результат": "ok", "Набор тестов": test_suite_name}
