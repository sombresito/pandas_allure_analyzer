"""Generate embeddings for Allure chunk files."""

from __future__ import annotations

from pathlib import Path
import os
import logging

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

# Paths can be overridden via environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "local_models/intfloat/multilingual-e5-small")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "chunks")
# Base directory for per-report embedding files
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR", "embeddings")

_MODEL = None


def get_model() -> SentenceTransformer:
    """Return cached SentenceTransformer instance."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_PATH)
    return _MODEL


def load_chunks(path: str | os.PathLike) -> pd.DataFrame:
    """Return a DataFrame with chunk data from ``path``.

    ``path`` may point to a single ``.jsonl`` file or a directory containing
    multiple chunk files.
    """
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("**/*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No .jsonl files found in {path}")
        dfs = [pd.read_json(f, lines=True) for f in files]
        return pd.concat(dfs, ignore_index=True)
    return pd.read_json(p, lines=True)


def create_embeddings(df: pd.DataFrame, model_path: str = MODEL_PATH) -> np.ndarray:
    """Generate embeddings for the ``rag_text`` column in ``df``."""
    if model_path != MODEL_PATH:
        model = SentenceTransformer(model_path)
    else:
        model = get_model()
    texts = df["rag_text"].tolist()
    return model.encode(
        texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )


def save_embeddings(
    embeddings: np.ndarray,
    team_name: str,
    report_uuid: str,
    base_dir: str | os.PathLike = EMBEDDINGS_DIR,
) -> Path:
    """Save ``embeddings`` for ``team_name``/``report_uuid`` and clean up old files.

    Returns the path where the embeddings were written.
    """
    dir_path = Path(base_dir) / team_name
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / f"{report_uuid}.npy"
    np.save(file_path, embeddings)

    files = sorted(dir_path.glob("*.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[3:]:
        old.unlink()
    return file_path


if __name__ == "__main__":
    df = load_chunks(CHUNKS_PATH)
    embeddings = create_embeddings(df)
    logger.info("Generated embeddings: %s", embeddings.shape)

    first_row = df.iloc[0]
    team = first_row["parentSuite"]
    report_uuid = first_row["report_uuid"]
    path = save_embeddings(embeddings, team, report_uuid)
    logger.info("Embeddings saved to %s", path)
