from fastapi import FastAPI, HTTPException, Request
import requests
import os
import logging
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
    extract_team_name,
    chunk_and_save_json,
    analyze_and_post,
    _auth_kwargs,
    ALLURE_API,
)
from embeddings import create_embeddings, load_chunks
from save_embeddings_to_qdrant import upload_embeddings
app = FastAPI()

@app.post("/uuid/analyze")
async def analyze_report(request: Request):
    body = await request.json()
    uuid = body.get("uuid")
    if not uuid:
        raise HTTPException(status_code=400, detail="UUID not provided.")
    logger.info("Analyze request received for UUID %s", uuid)

    # 1. Получаем JSON отчёт
    url = f"{ALLURE_API}/report/{uuid}/test-cases/aggregate"
    auth_kwargs = _auth_kwargs()
    try:
        resp = requests.get(url, verify=False, timeout=10, **auth_kwargs)
        resp.raise_for_status()
        logger.info("Fetched report data from Allure")
    except requests.RequestException as e:
        logger.error("Failed to fetch report for %s: %s", uuid, e)
        raise HTTPException(status_code=500, detail=f"Failed to fetch report: {e}") from e

    print(resp.status_code)
    print(repr(resp.text))
    
    try:
        report_data = resp.json()
    except ValueError as e:
        # сохраняем в файл для последующего анализа
        bad_path = f"/tmp/{uuid}_invalid_allure_response.txt"
        with open(bad_path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        logger.error("Invalid JSON received for %s, saved raw response to %s", uuid, bad_path)
        # пробрасываем понятную HTTP-ошибку
        raise HTTPException(
            status_code=502,
            detail=f"Invalid JSON received from Allure (see {bad_path})"
        )

    # 2. Получаем название команды
    team_name = extract_team_name(report_data)
    if not team_name:
        logger.error("Team name not found in report %s", uuid)
        raise HTTPException(status_code=400, detail="Team name (parentSuite) not found.")
    logger.info("Team name extracted: %s", team_name)

    # 3. Чанкуем и сохраняем
    json_path, df = chunk_and_save_json(report_data, uuid, team_name)
    logger.info("Chunks saved for %s", uuid)

    # 4. Генерация и загрузка эмбеддингов
    try:
        if df is None:
            df = load_chunks(json_path)
        embeddings = create_embeddings(df)
        upload_embeddings(df, embeddings, team_name, uuid)
        logger.info("Embeddings uploaded for %s", uuid)
    except Exception as e:
        logger.error("Failed to upload embeddings for %s: %s", uuid, e)
        raise HTTPException(status_code=500, detail=f"Failed to upload embeddings: {e}") from e

    # 5. Анализ и отправка результата
    try:
        analyze_and_post(uuid, team_name, report_data)
        logger.info("Analysis posted for %s", uuid)
    except Exception as e:
        logger.error("Analysis failed for %s: %s", uuid, e)
        return {"result": "partial", "error": str(e)}

    return {"result": "ok", "team": team_name}
