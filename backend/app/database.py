from collections.abc import Generator
import logging
import os
from pathlib import Path
import socket
from urllib.parse import urlparse

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

logger = logging.getLogger(__name__)

_DEV_LOCAL_FALLBACK_ENVS = frozenset({"development", "dev", "local"})
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _is_tcp_open(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _resolve_database_url(raw_url: str, environment: str) -> str:
    """Fall back to local SQLite in dev when localhost PostgreSQL is stopped."""
    env_norm = (environment or "").strip().lower()
    if (
        env_norm in _DEV_LOCAL_FALLBACK_ENVS
        and raw_url.startswith(("postgresql", "postgres"))
        and os.environ.get("ARES_REQUIRE_POSTGRES", "0") != "1"
    ):
        parsed = urlparse(raw_url)
        host = (parsed.hostname or "").lower()
        port = parsed.port or 5432
        if host in _LOCAL_HOSTS and not _is_tcp_open(host, port):
            sqlite_path = Path(__file__).resolve().parent.parent / "ares.db"
            logger.warning(
                "Local PostgreSQL at %s:%d is unreachable in %s mode; falling back to SQLite at %s",
                host,
                port,
                environment,
                sqlite_path,
            )
            return f"sqlite:///{sqlite_path.as_posix()}"
    return raw_url


settings = get_settings()
resolved_database_url = _resolve_database_url(settings.database_url, settings.environment)
connect_args = {"check_same_thread": False} if resolved_database_url.startswith("sqlite") else {}

# Add connection timeout for PostgreSQL
if resolved_database_url.startswith("postgresql") or resolved_database_url.startswith("postgres"):
    connect_args["connect_timeout"] = 5

# Configure connection pool for production databases
if resolved_database_url.startswith("sqlite"):
    engine = create_engine(
        resolved_database_url,
        pool_pre_ping=True,
        connect_args=connect_args,
    )
else:
    # PostgreSQL or other production databases
    engine = create_engine(
        resolved_database_url,
        pool_pre_ping=True,
        pool_size=20,
        max_overflow=40,
        pool_recycle=3600,
        pool_timeout=30,
        connect_args=connect_args,
    )
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
