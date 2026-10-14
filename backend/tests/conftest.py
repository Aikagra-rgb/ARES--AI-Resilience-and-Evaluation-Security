import os
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Test-suite environment. These must be set BEFORE any `ares`/`app` module is imported, because both
# settings objects are built at import time. Production-grade defaults (auth on, no header-only
# identity) are exercised explicitly in test_security.py by patching the live settings objects.
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("ARES_AUTH_ENABLED", "false")
os.environ.setdefault("ALLOW_DEV_IDENTITY", "true")
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://postgres:postgres@127.0.0.1:5432/ares_test")

# Add backend directory to sys.path so 'ares' can be imported
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Ensure database schema is created on test database if PostgreSQL is available."""
    database_url = os.environ.get("DATABASE_URL", "postgresql+psycopg://postgres:postgres@127.0.0.1:5432/ares_test")
    try:
        engine = create_engine(database_url, connect_args={"connect_timeout": 3})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        import app.models  # Ensure all SQLAlchemy models are registered
        from app.database import Base
        Base.metadata.create_all(bind=engine)
        yield
        engine.dispose()
    except Exception:
        # PostgreSQL unavailable in local dev environment
        yield


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "integration: mark test as requiring live external API credentials")
    config.addinivalue_line("markers", "slow: mark test as slow running")