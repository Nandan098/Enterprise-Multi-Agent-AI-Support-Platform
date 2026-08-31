from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from rag.embeddings import OUTPUT_DIMENSIONALITY, embed_query

INDEX_DIR = Path(__file__).resolve().parents[1] / "data" / "faiss_index"
INDEX_FILE = INDEX_DIR / "index.faiss"
META_FILE = INDEX_DIR / "metadata.json"


def index_exists() -> bool:
    return INDEX_FILE.exists() and META_FILE.exists()


def _load() -> tuple[faiss.Index, list[dict]] | None:
    if not index_exists():
        return None

    index = faiss.read_index(str(INDEX_FILE))

    if index.d != OUTPUT_DIMENSIONALITY:
        raise RuntimeError(
            f"FAISS index dimension {index.d} does not match "
            f"{OUTPUT_DIMENSIONALITY}."
        )

    with META_FILE.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    if index.ntotal != len(metadata):
        raise RuntimeError(
            "FAISS index and metadata are out of sync."
        )

    return index, metadata


def retrieve_with_scores(question: str, k: int = 6) -> list[dict]:
    loaded = _load()

    if loaded is None:
        return []

    index, metadata = loaded

    query_vector = np.asarray(
        [embed_query(question)],
        dtype="float32",
    )

    k = min(k, index.ntotal)

    distances, ids = index.search(query_vector, k)

    hits = []

    for distance, item_id in zip(
        distances[0].tolist(),
        ids[0].tolist(),
    ):
        if item_id < 0:
            continue

        item = metadata[item_id]

        hits.append(
            {
                "content": item["content"],
                "source": item.get("source", "unknown"),
                "page": item.get("page", 1),
                "chunk_id": item.get("chunk_id", "unknown"),
                "distance": round(float(distance), 4),
            }
        )

    return hits