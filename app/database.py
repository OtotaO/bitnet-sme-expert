"""Database utilities and session management."""
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
import os

# Database URL - can be overridden by environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./bitnet_sme.db")

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

def init_db():
    """Initialize the database."""
    from .models.training import Base as TrainingBase
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Import models to ensure they are registered with SQLAlchemy
    from .models import training  # noqa

@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_session() -> Session:
    """Get a database session (for use with FastAPI Depends)."""
    with get_db() as db:
        yield db
