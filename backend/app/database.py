from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

settings = get_settings()
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

# Add connection timeout for PostgreSQL
if settings.database_url.startswith("postgresql") or settings.database_url.startswith("postgres"):
    connect_args["connect_timeout"] = 5

# Configure connection pool for production databases
if settings.database_url.startswith("sqlite"):
    engine = create_engine(
        settings.database_url, 
        pool_pre_ping=True, 
        connect_args=connect_args
    )
else:
    # PostgreSQL or other production databases
    engine = create_engine(
        settings.database_url,
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
