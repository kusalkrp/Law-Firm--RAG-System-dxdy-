"""End-to-end pipeline tests for legal query answering and hallucination guards."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_in_scope_query_generates_cited_answer():
    """Verify in-scope legal query returns grounded answer with citations and high faithfulness."""
    res = client.post("/query", json={"question": "What are the indemnification obligations in the Manufacturing Agreement?"})
    assert res.status_code == 200
    data = res.json()
    assert len(data["citations"]) > 0
    assert data["faithfulness_score"] is not None
    assert data["confidence"] == "high"
    assert data["abstention_triggered_by"] is None
    assert "Manufacturing Agreement" in data["citations"][0]["source_file"]


def test_out_of_scope_query_abstains():
    """Verify out-of-scope query triggers abstention without hallucinating."""
    res = client.post("/query", json={"question": "What is the capital city of France and its population?"})
    assert res.status_code == 200
    data = res.json()
    assert "cannot find this information" in data["answer"].lower()
    assert data["abstention_triggered_by"] in ["pre_check", "model"]
    assert len(data["citations"]) == 0


def test_absent_entity_query_abstains():
    """Verify query with entities absent from contract corpus triggers abstention."""
    res = client.post("/query", json={"question": "What is the stock ticker for Tesla in the Trademark Agreement?"})
    assert res.status_code == 200
    data = res.json()
    assert "cannot find this information" in data["answer"].lower()
    assert data["abstention_triggered_by"] in ["pre_check", "model"]
