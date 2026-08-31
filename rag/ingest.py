from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from pypdf import PdfReader

from rag.embeddings import OUTPUT_DIMENSIONALITY, embed_documents

INDEX_DIR = Path(__file__).resolve().parents[1] / "data" / "faiss_index"
INDEX_FILE = INDEX_DIR / "index.faiss"
META_FILE = INDEX_DIR / "metadata.json"


def _split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    text = " ".join((text or "").split())
    if not text:
        return []

    chunks: list[str] = []
    step = max(1, chunk_size - overlap)

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)

    return chunks

def ingest_pdf(pdf_path: Path | str) -> dict[str, Any]:
    """
    Ingest a PDF using pypdf + Gemini embeddings + raw FAISS.
    This avoids LangChain's PDF loader and the xxhash native dependency.
    """
    
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))

    records: list[dict[str, Any]] = []

    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        for local_idx, chunk_text in enumerate(_split_text(text)):
            records.append(
                {
                    "content": chunk_text,
                    "source": pdf_path.name,
                    "page": page_idx + 1,
                    "chunk_id": f"{pdf_path.stem}-{page_idx + 1}-{local_idx}",
                }
            )

    if not records:
        raise ValueError(
            f"No extractable text found in PDF: {pdf_path.name}"
        )

    vectors = np.asarray(
        embed_documents([r["content"] for r in records]),
        dtype="float32",
    )

    if vectors.ndim != 2 or vectors.shape[1] != OUTPUT_DIMENSIONALITY:
        raise RuntimeError(
            f"Unexpected embedding shape {vectors.shape}; "
            f"expected (*, {OUTPUT_DIMENSIONALITY})."
        )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if INDEX_FILE.exists() and META_FILE.exists():
        index = faiss.read_index(str(INDEX_FILE))

        with META_FILE.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        if index.d != OUTPUT_DIMENSIONALITY:
            raise RuntimeError(
                f"Existing FAISS index dimension {index.d} "
                f"does not match {OUTPUT_DIMENSIONALITY}."
            )
    else:
        index = faiss.IndexFlatL2(OUTPUT_DIMENSIONALITY)
        metadata = []

    index.add(vectors)
    metadata.extend(records)

    faiss.write_index(index, str(INDEX_FILE))

    with META_FILE.open("w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "pages": len(reader.pages),
        "chunks": len(records),
        "index_path": str(INDEX_DIR),
    }