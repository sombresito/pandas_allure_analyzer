import requests
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
try:
    from qdrant_client.http.exceptions import UnexpectedResponse, QdrantConnectionError as QdrantClientConnectionError
except Exception:
    class UnexpectedResponse(Exception):
        pass

    class QdrantClientConnectionError(Exception):
        pass
import json
import os
import logging

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

# ==== Настройки ====
MODEL_PATH = "local_models/intfloat/multilingual-e5-small"
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "10"))
COLLECTION_NAME = "allure_chunks"
# URL for the Ollama API can be overridden by environment variable
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434/api/generate")
OLLAMA_MODEL = "mistral:latest"

# Default analysis prompt can be overridden via the /prompt API
question = (
    "Проанализируй текущий отчёт (выводы, обратная связь, рекомендации), "
    "а затем сравни с двумя предыдущими и укажи тренд."
)


# ==== Инициализация ====
_MODEL = None
_CLIENT = None


class QdrantConnectionError(Exception):
    """Raised when a connection to Qdrant cannot be established."""


class RagAnalysisError(Exception):
    """Raised when analysis fails due to Qdrant errors."""


def get_model():
    """Return cached SentenceTransformer instance."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_PATH)
    return _MODEL


def get_client():
    """Return cached QdrantClient instance."""
    global _CLIENT
    if _CLIENT is None:
        try:
            _CLIENT = QdrantClient(
                url=QDRANT_URL,
                prefer_grpc=False,
                timeout=QDRANT_TIMEOUT,
                check_compatibility=False,
            )
        except Exception as e:  # pragma: no cover - network errors hard to simulate
            logger.error("Failed to connect to Qdrant at %s: %s", QDRANT_URL, e)
            raise QdrantConnectionError(
                f"Could not connect to Qdrant at {QDRANT_URL}"
            ) from e
    return _CLIENT


# ==== Поиск релевантных векторов ====
def search_similar_chunks(query: str, top_k: int = 5):
    model = get_model()
    client = get_client()
    query_embedding = model.encode(query, convert_to_numpy=True).tolist()
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
    )
    return [hit.payload["rag_text"] for hit in results]


# ==== Генерация ответа через Ollama ====
def generate_answer_with_ollama(
    chunks,
    question,
    ollama_url: str = OLLAMA_URL,
    prompt_override: str | None = None,
):
    context = "\n\n".join(chunks)
    if prompt_override is not None:
        prompt = f"Вот информация из отчёта:\n{context}\n\n{prompt_override}"
    else:
        prompt = (
            f"Вот информация из отчёта:\n{context}\n\n"
            f"Вопрос: {question}\n\n"
            "1) Проанализируй **текущий** отчёт: выведи ключевые выводы и метрики, дай подробную обратную связь и рекомендации по улучшению следующих прогонов.\n"
            "2) Затем сравни этот отчёт с двумя предыдущими отчётами команды и определи тренд: деградация, улучшение или стабильность.\n\n"
            "Ответ структурируй по пунктам 1. Анализ текущего очтета: и 2. Сравнение с двумя предыдущими отчётами:, но не пиши цифры, названия пунктов сделай жирным, а текст — обычным."
        )
    logger.info("Промпт который получает ИИ: %s", prompt)
    try:
        response = requests.post(
            ollama_url,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "temperature": "0.3",
                "stream": True
            },
            stream=True,
            timeout=300
        )
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error("Failed to generate answer via Ollama: %s", e)
        raise RagAnalysisError("Failed to generate answer") from e

    answer = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            answer += data.get("response", "")

    return answer.strip()


def run_rag_analysis(
    test_suite_name: str,
    question_override: str | None = None,
    prompt_override: str | None = None,
) -> dict:
    """Generate a short analysis for a team's latest report using RAG.

    Parameters
    ----------
    test_suite_name: str
        Team name used as filter when fetching chunks from Qdrant.
    question_override: str | None, optional
        Prompt to use instead of the default :data:`question`.  When ``None``
        the globally configured prompt is used.
    """
    client = get_client()
    search_filter = Filter(
        must=[FieldCondition(key="team", match=MatchValue(value=test_suite_name))]
    )
    try:
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=search_filter,
            with_payload=True,
            limit=1000,
        )
    except (UnexpectedResponse, QdrantClientConnectionError) as e:  # pragma: no cover - network errors hard to simulate
        logger.error("Failed to retrieve chunks from Qdrant: %s", e)
        raise RagAnalysisError("Qdrant unreachable or returned an error") from e
    chunks = [p.payload.get("rag_text", "") for p in points]

    q = question_override or question
    logger.info("Овверайд промпта: %s", q)
    answer = (
        generate_answer_with_ollama(chunks, q, prompt_override=prompt_override)
        if chunks
        else ""
    )
    return {"team": test_suite_name, "analysis": answer}


# ==== Основная функция ====
def ask(question: str):
    logger.info("[Q] %s", question)
    chunks = search_similar_chunks(question, top_k=5)
    answer = generate_answer_with_ollama(chunks, question)
    logger.info("[A] %s", answer)


# ==== Пример запуска ====
if __name__ == "__main__":
    ask("Какие тесты завершились с ошибкой?")
