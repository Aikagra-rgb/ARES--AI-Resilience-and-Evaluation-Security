"""Tests for local development PostgreSQL-to-SQLite fallback in app.database."""

from __future__ import annotations

import pytest

from app.database import _resolve_database_url


def test_resolve_database_url_falls_back_in_dev_when_localhost_pg_closed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ARES_REQUIRE_POSTGRES", raising=False)
    monkeypatch.setattr("app.database._is_tcp_open", lambda host, port, timeout=0.3: False)

    resolved = _resolve_database_url("postgresql+psycopg://postgres@127.0.0.1:5432/ares", "development")
    assert resolved.startswith("sqlite:///")
    assert resolved.endswith("/backend/ares.db")


@pytest.mark.parametrize("env_name", ["production", "staging", "test", "testing"])
def test_resolve_database_url_never_falls_back_outside_local_dev(
    monkeypatch: pytest.MonkeyPatch, env_name: str
):
    monkeypatch.delenv("ARES_REQUIRE_POSTGRES", raising=False)
    monkeypatch.setattr("app.database._is_tcp_open", lambda host, port, timeout=0.3: False)

    pg_url = "postgresql+psycopg://postgres@127.0.0.1:5432/ares"
    assert _resolve_database_url(pg_url, env_name) == pg_url


def test_resolve_database_url_respects_require_postgres_and_remote_hosts(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("app.database._is_tcp_open", lambda host, port, timeout=0.3: False)

    local_pg = "postgresql+psycopg://postgres@127.0.0.1:5432/ares"
    monkeypatch.setenv("ARES_REQUIRE_POSTGRES", "1")
    assert _resolve_database_url(local_pg, "development") == local_pg

    monkeypatch.delenv("ARES_REQUIRE_POSTGRES", raising=False)
    remote_pg = "postgresql+psycopg://ares:pass@db.internal:5432/ares"
    assert _resolve_database_url(remote_pg, "development") == remote_pg
