import json
from pathlib import Path

import pytest


def test_golden_dataset_exists():
    data = json.loads((Path("evaluation") / "golden_questions.json").read_text())
    assert len(data) >= 3
    assert all("expected_route" in x for x in data)


def test_application_imports_when_dependencies_are_installed():
    try:
        from agents.graph import build_graph
        from rag.ingest import ingest_pdf
        from rag.retriever import retrieve_with_scores
        from tools.actions import create_support_ticket
    except ModuleNotFoundError as exc:
        pytest.skip(f"Runtime dependency not installed in this environment: {exc}")
    assert build_graph() is not None
    assert callable(ingest_pdf)
    assert callable(retrieve_with_scores)
    assert create_support_ticket("VPN")["status"] == "CREATED"
