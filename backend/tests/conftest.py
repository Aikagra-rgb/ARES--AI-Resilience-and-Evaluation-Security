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

# Add backend directory to sys.path so 'ares' can be imported
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))


@pytest.fixture(scope="session")
def pg_engine():
    """Create a PostgreSQL engine for the test session."""
    # Use the DATABASE_URL from environment or default to local test database
    database_url = os.environ.get("DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/ares_test")
    engine = create_engine(database_url)
    
    # Create all tables
    from app.database import Base
    Base.metadata.create_all(bind=engine)
    
    yield engine
    
    # Clean up
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(pg_engine):
    """Create a new database session for each test, truncating tables after."""
    connection = pg_engine.connect()
    transaction = connection.begin()
    
    SessionLocal = sessionmaker(bind=connection)
    session = SessionLocal()
    
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture(autouse=True)
def truncate_tables(db_session):
    """Truncate all tables after each test to ensure isolation."""
    yield
    # Truncate all tables in reverse dependency order
    from app.models import (
        Finding, TestRun, Evidence, Policy, Project, 
        User, Organization, ArenaSubmission
    )
    tables = [
        Finding, TestRun, Evidence, Policy, Project,
        User, Organization, ArenaSubmission
    ]
    for table in tables:
        try:
            db_session.execute(text(f"TRUNCATE TABLE {table.__tablename__} RESTART IDENTITY CASCADE"))
        except Exception:
            pass  # Table might not exist
    db_session.commit()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "integration: mark test as requiring live external API credentials")
    config.addinivalue_line("markers", "slow: mark test as slow running")