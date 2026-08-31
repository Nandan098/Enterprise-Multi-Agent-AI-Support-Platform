# Enterprise Multi-Agent AI Support Platform

A Cognizant Ace Frontier-oriented portfolio project that demonstrates RAG, multi-agent orchestration, controlled tool use, AI validation, evaluation, observability, and deployment.

## What this project demonstrates
- Enterprise PDF ingestion with metadata-aware chunking
- FAISS vector retrieval with source/page/chunk citations
- LangGraph orchestration with Router, Knowledge, Support, Action, and Validator agents
- Controlled support-ticket tool calling (demo in-memory action)
- Grounding validation with safe fallback when evidence is insufficient
- Streamlit enterprise-style UI
- FastAPI backend
- Curated evaluation dataset and regression-style evaluator
- Structured operational metrics: route, retrieval count, validation, latency
- Docker and GitHub Actions CI workflow

## Architecture

```text
                         +--------------------+
                         |   Streamlit UI     |
                         +---------+----------+
                                   |
                              HTTP / REST
                                   |
                         +---------v----------+
                         |     FastAPI        |
                         +---------+----------+
                                   |
                         +---------v----------+
                         | LangGraph Router   |
                         +---+------+------+-+
                             |      |      |
                         KNOWLEDGE SUPPORT ACTION
                             |      |      |
                             +------+------+
                                    |
                          +---------v----------+
                          |     Validator      |
                          +---------+----------+
                                    |
                         Grounded final answer
                                    |
             +----------------------+--------------------+
             |                                           |
      +------v------+                            +--------v-------+
      | FAISS index |                            | Demo REST tool |
      +-------------+                            +----------------+
```

## Stack
Python, FastAPI, Streamlit, LangGraph, LangChain, Google Gemini, FAISS, PyPDF, Pydantic, Docker, GitHub Actions.

## Requirements
Python 3.11+ is recommended.

Set the environment:

```bash
cp .env.example .env
# Put your API key in .env
```

Install:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Run API:

```bash
uvicorn backend.main:app --reload
```

In another terminal:

```bash
streamlit run frontend/app.py
```

Open the Streamlit URL shown by the command.

## First demo

1. Start API and Streamlit.
2. Upload `data/sample_enterprise_policy.pdf` from the sidebar.
3. Click **Index document**.
4. Ask: `What is the password reset policy?`
5. Show the route, validator result, source/page/chunk citations, and retrieved excerpts.
6. Ask: `Create a support ticket for my VPN issue.` to show controlled action execution.
7. Run the evaluation suite from a terminal:

```bash
python -m evaluation.run_eval
```

## Interview story

The core story is: **I did not build a PDF chatbot; I built a small enterprise AI platform.**

The system routes work between specialized agents, grounds knowledge answers in retrieved evidence, validates the generated response before release, exposes source metadata, and measures basic operational quality. The action agent demonstrates how an LLM can invoke a controlled tool instead of directly performing arbitrary external actions.

## Evaluation

`evaluation/golden_questions.json` contains a small curated benchmark. The evaluator checks:
- expected route
- expected keywords in the answer
- validator status
- latency

The benchmark is deliberately small for a one-week portfolio build. In an interview, describe the natural next step as a larger golden set plus automated faithfulness/relevance metrics and CI regression thresholds.

## Observability

The application emits JSON logs for request start/end, selected route, retrieval count, validation status, and latency. This is intentionally lightweight and easy to understand in an interview. OpenTelemetry can be added as a next production hardening step.

## Safety / guardrails

- Knowledge/support responses are generated only from retrieved context.
- The validator checks grounding before the response is released.
- If grounding is insufficient, the system refuses to answer rather than inventing facts.
- Action execution is isolated behind an explicit tool function; the LLM does not receive arbitrary code execution access.

## Project structure

```text
cognizant_ace_project1_final/
├── agents/graph.py
├── backend/main.py
├── frontend/app.py
├── rag/ingest.py
├── rag/retriever.py
├── tools/actions.py
├── evaluation/golden_questions.json
├── evaluation/run_eval.py
├── observability/logging_config.py
├── tests/test_project.py
├── data/sample_enterprise_policy.pdf
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
├── .env.example
├── requirements.txt
└── README.md
```


## Public demo deployment

The Streamlit frontend is designed for a zero-setup demo. The FastAPI backend automatically indexes the bundled `data/sample_enterprise_policy.pdf` on startup, so a visitor can open the Streamlit URL and immediately ask questions.

For cloud deployment, run the FastAPI service separately (for example on Render) and set `BACKEND_URL` in Streamlit Cloud Secrets to the public FastAPI URL. Set `GEMINI_API_KEY` as a secret on the backend.
