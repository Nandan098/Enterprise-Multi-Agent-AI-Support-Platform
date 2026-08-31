from __future__ import annotations

import os
from typing import Iterable

from dotenv import load_dotenv
load_dotenv()

from google import genai

OUTPUT_DIMENSIONALITY = int(os.getenv("EMBEDDING_DIM", "768"))
MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def embed_documents(texts: Iterable[str]) -> list[list[float]]:
    texts = list(texts)

    if not texts:
        return []

    result = _client.models.embed_content(
        model=MODEL,
        contents=texts,
        config={
            "output_dimensionality": OUTPUT_DIMENSIONALITY,
        },
    )

    return [embedding.values for embedding in result.embeddings]


def embed_query(text: str) -> list[float]:
    result = _client.models.embed_content(
        model=MODEL,
        contents=text,
        config={
            "output_dimensionality": OUTPUT_DIMENSIONALITY,
        },
    )

    return result.embeddings[0].values