import os
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import pytest
from sqlalchemy import create_engine, text

# Add backend directory to sys.path so 'ares' can be imported
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))


def _is_tcp_open(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


_PYTEST_SQLITE_FILE = backend_dir / ".pytest_ares_test.db"

# Test-suite environment. These must be set BEFORE any `ares`/`app` module is imported, because both
# settings objects are built at import time. Production-grade defaults (auth on, no header-only
# identity) are exercised explicitly in test_security.py by patching the live settings objects.
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("ARES_AUTH_ENABLED", "false")
os.environ.setdefault("ALLOW_DEV_IDENTITY", "true")
os.environ.setdefault("RUN_WORKER", "false")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("EXECUTION_PAYLOAD_ENCRYPTION_KEY", "Jz7cmIiWycPoeJ4u0Vcg5n_lDrU9pCqQR81hP25wUco=")

_configured_db = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@127.0.0.1:5432/ares_test",
)
if _configured_db.startswith("postgresql") and os.environ.get("ARES_REQUIRE_POSTGRES", "0") != "1":
    _parsed_db = urlparse(_configured_db)
    _db_host = _parsed_db.hostname or "127.0.0.1"
    _db_port = _parsed_db.port or 5432
    if _db_host in ("127.0.0.1", "localhost", "::1") and not _is_tcp_open(_db_host, _db_port):
        try:
            if _PYTEST_SQLITE_FILE.exists():
                _PYTEST_SQLITE_FILE.unlink()
        except OSError:
            pass
        os.environ["DATABASE_URL"] = f"sqlite:///{_PYTEST_SQLITE_FILE.as_posix()}"
        os.environ["AUTO_CREATE_SCHEMA"] = "true"
    else:
        os.environ.setdefault("DATABASE_URL", _configured_db)
else:
    os.environ.setdefault("DATABASE_URL", _configured_db)


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Ensure database schema is created on PostgreSQL or local SQLite test fallback."""
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@127.0.0.1:5432/ares_test",
    )

    if database_url.startswith("sqlite"):
        engine = create_engine(database_url, connect_args={"check_same_thread": False})
        import app.models  # noqa: F401 - ensure all SQLAlchemy models are registered
        from app.database import Base

        Base.metadata.create_all(bind=engine)
        try:
            yield
        finally:
            engine.dispose()
            try:
                from app.database import engine as app_engine

                app_engine.dispose()
            except Exception:
                pass
            try:
                if _PYTEST_SQLITE_FILE.exists():
                    _PYTEST_SQLITE_FILE.unlink()
            except OSError:
                pass
        return

    # If the target test database doesn't exist yet, automatically create it via maintenance database
    try:
        parsed = urlparse(database_url)
        target_db = parsed.path.lstrip("/")
        if target_db and target_db != "postgres":
            maintenance_url = urlunparse(parsed._replace(path="/postgres"))
            m_engine = create_engine(maintenance_url, isolation_level="AUTOCOMMIT", connect_args={"connect_timeout": 3})
            with m_engine.connect() as m_conn:
                exists = m_conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :dbname"), {"dbname": target_db}).scalar()
                if not exists:
                    m_conn.execute(text(f'CREATE DATABASE "{target_db}"'))
            m_engine.dispose()
    except Exception:
        pass

    try:
        engine = create_engine(database_url, connect_args={"connect_timeout": 3})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        import app.models  # noqa: F401 - ensure all SQLAlchemy models are registered
        from app.database import Base
        Base.metadata.create_all(bind=engine)
        yield
        engine.dispose()
    except Exception:
        # PostgreSQL unavailable in local dev environment
        yield


@pytest.fixture(autouse=True)
def reset_app_rate_limiter():
    """Clear in-memory rate-limit windows between tests to prevent cross-test 429 bleed."""
    def _clear() -> None:
        main_mod = sys.modules.get("app.main")
        if main_mod is None:
            return
        limiter = getattr(main_mod, "rate_limiter", None)
        if limiter is None:
            return
        target = getattr(limiter, "_fallback", limiter)
        windows = getattr(target, "_windows", None)
        lock = getattr(target, "_lock", None)
        if isinstance(windows, dict):
            if lock is not None:
                with lock:
                    windows.clear()
            else:
                windows.clear()

    _clear()
    yield
    _clear()


@pytest.fixture(autouse=True)
def reset_circuit_breakers():
    """Clear global circuit-breaker registry between tests to prevent cross-test OPEN bleed."""
    def _clear() -> None:
        cb_mod = sys.modules.get("ares.security.circuit_breaker")
        if cb_mod is None:
            return
        breakers = getattr(cb_mod, "_BREAKERS", None)
        lock = getattr(cb_mod, "_REGISTRY_LOCK", None)
        if isinstance(breakers, dict):
            if lock is not None:
                with lock:
                    breakers.clear()
            else:
                breakers.clear()

    _clear()
    yield
    _clear()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "integration: mark test as requiring live external API credentials")
    config.addinivalue_line("markers", "slow: mark test as slow running")