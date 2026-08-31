from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

from agents.graph import run_agent
from observability.logging_config import logger
from rag.ingest import ingest_pdf
from rag.retriever import index_exists

app = FastAPI(title="Enterprise Multi-Agent AI Support Platform", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)
DEMO_PDF = DATA_DIR / "sample_enterprise_policy.pdf"

class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)

class EvaluateRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    expected_keywords: list[str] = Field(default_factory=list)

def ensure_demo_index() -> None:
    if index_exists():
        return
    if DEMO_PDF.exists():
        logger.info("demo_index_build_start")
        result = ingest_pdf(DEMO_PDF)
        logger.info(
            "demo_index_build_complete",
            extra={"pages": result.get("pages"), "chunks": result.get("chunks")},
        )

@app.on_event("startup")
def startup() -> None:
    # Make the public demo zero-setup: the bundled sample knowledge base is
    # indexed automatically when the backend starts.
    try:
        ensure_demo_index()
    except Exception:
        logger.exception("demo_index_build_failed")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ace-frontier-project1",
        "demo_indexed": index_exists(),
    }

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = Path(file.filename or "document.pdf").name
    if not filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are supported."}
    target = DATA_DIR / filename
    target.write_bytes(await file.read())
    result = ingest_pdf(target)
    return {"filename": filename, **result}

@app.post("/chat")
def chat(payload: ChatRequest) -> dict[str, Any]:
    ensure_demo_index()
    started = time.perf_counter()
    result = run_agent(payload.question)
    latency = round((time.perf_counter() - started) * 1000, 2)
    result.update({"latency_ms": latency, "retrieval_count": len(result.get("retrieval", []))})
    logger.info(
        "chat_complete",
        extra={
            "route": result.get("route"),
            "latency_ms": latency,
            "retrieval_count": result.get("retrieval_count"),
            "validation": result.get("validation"),
        },
    )
    return result

@app.post("/evaluate")
def evaluate(payload: EvaluateRequest) -> dict[str, Any]:
    ensure_demo_index()
    started = time.perf_counter()
    result = run_agent(payload.question)
    answer = result.get("answer", "").lower()
    keywords = [k.lower().strip() for k in payload.expected_keywords if k.strip()]
    matched = [k for k in keywords if k in answer]
    score = round(len(matched) / len(keywords), 3) if keywords else None
    latency = round((time.perf_counter() - started) * 1000, 2)
    return {
        **result,
        "keyword_score": score,
        "matched_keywords": matched,
        "latency_ms": latency,
    }
