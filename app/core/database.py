"""
Database Configuration — SQLAlchemy engine, session factory, Base, get_db.

Interface-compatible with existing app.core.database so all engine imports
(from app.core.database import Base, get_db) resolve correctly.

Default: SQLite (zero config)
Production: set DATABASE_URL=postgresql://user:pass@host:5432/dbname
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings

connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    echo=os.getenv("SQL_ECHO", "false").lower() == "true",
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Yield a DB session per request. Auto-closes on completion."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
