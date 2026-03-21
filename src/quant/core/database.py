"""Database engine and session management."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from quant.core.config import Settings


def get_engine(settings: Settings):
    return create_engine(settings.database.url, echo=False)


def get_session_factory(settings: Settings) -> sessionmaker[Session]:
    engine = get_engine(settings)
    return sessionmaker(bind=engine)


def init_db(settings: Settings):
    """Create all tables."""
    from quant.core.models import Base

    engine = get_engine(settings)
    Base.metadata.create_all(engine)
