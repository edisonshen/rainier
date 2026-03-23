"""Database engine, session factory, and initialization."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from rainier.core.config import Settings, get_settings
from rainier.core.models import HYPERTABLES, Base

log = structlog.get_logger()

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine(settings: Settings | None = None) -> Engine:
    """Get or create the SQLAlchemy engine (singleton)."""
    global _engine
    if _engine is None:
        if settings is None:
            settings = get_settings()
        _engine = create_engine(
            settings.database_url,
            echo=settings.database.echo,
            pool_size=settings.database.pool_size,
            pool_pre_ping=True,
        )
    return _engine


def _get_session_factory() -> sessionmaker[Session]:
    """Get or create the session factory (singleton)."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _session_factory


def get_session_factory(settings: Settings) -> sessionmaker[Session]:
    """Backward-compatible session factory (takes settings arg)."""
    engine = get_engine(settings)
    return sessionmaker(bind=engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a database session that auto-commits on success, rolls back on error."""
    session = _get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _create_hypertables(engine: Engine) -> None:
    """Convert time-series tables to TimescaleDB hypertables (idempotent)."""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))
        conn.commit()

        for table_name, time_column in HYPERTABLES.items():
            try:
                conn.execute(
                    text(
                        f"SELECT create_hypertable('{table_name}', '{time_column}', "
                        f"migrate_data => true, if_not_exists => true)"
                    )
                )
                conn.commit()
                log.info("hypertable_created", table=table_name, time_column=time_column)
            except Exception as exc:
                conn.rollback()
                log.warning(
                    "hypertable_skipped",
                    table=table_name,
                    reason=str(exc),
                )


def init_db() -> None:
    """Create all tables and set up TimescaleDB hypertables."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    log.info("tables_created", tables=list(Base.metadata.tables.keys()))
    _create_hypertables(engine)
    log.info("database_initialized")
