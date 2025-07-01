"""Utility for uploading embeddings to Qdrant."""

import os
import numpy as np
import pandas as pd
import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PointIdsList,
)

from embeddings import load_chunks

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

# Настройки
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "10"))
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    prefer_grpc=False,
    timeout=QDRANT_TIMEOUT,
    check_compatibility=False,
)
COLLECTION_NAME = "allure_chunks"


def ensure_collection(client: QdrantClient, vector_size: int) -> None:
    """Create the collection if it doesn't exist."""
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info("Collection '%s' created", COLLECTION_NAME)


def upload_embeddings(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    team: str,
    report_uuid: str,
    client: QdrantClient | None = None,
) -> None:
    """Upload embeddings for a single report to Qdrant."""

    if client is None:
        client = qdrant_client

    ensure_collection(client, embeddings.shape[1])

    search_filter = Filter(must=[FieldCondition(key="team", match=MatchValue(value=team))])
    existing = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=search_filter,
        limit=1000,
        with_payload=True,
    )[0]

    from collections import defaultdict

    by_uuid = defaultdict(list)
    for point in existing:
        report_id = point.payload.get("report_uuid", "unknown")
        by_uuid[report_id].append(point)

    if len(by_uuid) >= 3:
        sorted_uuids = sorted(by_uuid.items(), key=lambda x: x[0])
        for old_uuid, old_points in sorted_uuids[:-2]:
            ids_to_delete = [p.id for p in old_points]
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=PointIdsList(points=ids_to_delete),
            )
            logger.info("Removed old report: %s", old_uuid)

    points = [
        PointStruct(
            id=report_uuid,
            vector=embeddings[i],
            payload={
                "rag_text": df.loc[i, "rag_text"],
                "name": df.loc[i, "name"],
                "status": df.loc[i, "status"],
                "suite": df.loc[i, "suite"],
                "uid": df.loc[i, "uid"],
                "team": team,
                "report_uuid": report_uuid,
            },
        )
        for i in range(len(df))
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.info("Uploaded %s points for team '%s' and UUID %s", len(points), team, report_uuid)

if __name__ == "__main__":
    CHUNKS_PATH = os.getenv("CHUNKS_PATH", "chunks")
    EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR", "embeddings")

    df = load_chunks(CHUNKS_PATH)
    first_row = df.iloc[0]
    team = first_row.get("parentSuite") or first_row.get("suite")
    report_uuid = first_row["report_uuid"]

    emb_path = os.path.join(EMBEDDINGS_DIR, team, f"{report_uuid}.npy")
    embeddings = np.load(emb_path)

    upload_embeddings(df, embeddings, team, report_uuid)
