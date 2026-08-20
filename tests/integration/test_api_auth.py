"""HTTP-level coverage for the optional web-to-sidecar shared secret."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import unmark.api.app as api_app


def test_configured_api_token_protects_health_and_rejects_bad_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_app, "_API_TOKEN", "test-sidecar-secret")

    with TestClient(api_app.create_app()) as client:
        missing = client.get("/health")
        wrong = client.get("/health", headers={"Authorization": "Bearer wrong"})
        correct = client.get(
            "/health",
            headers={"Authorization": "Bearer test-sidecar-secret"},
        )

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert correct.status_code == 200
    assert correct.json()["status"] == "ok"


def test_unconfigured_api_token_leaves_local_development_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_app, "_API_TOKEN", None)

    with TestClient(api_app.create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
