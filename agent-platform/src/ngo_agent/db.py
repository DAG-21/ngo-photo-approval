"""Engine and session factory. PostgreSQL in production (DATABASE_URL), SQLite for tests and the local demo."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from .models import Base
from .triggers import install_triggers

DEFAULT_URL = "sqlite:///./ngo_agent.db"


def make_engine(url: Optional[str] = None, echo: bool = False) -> Engine:
    url = url or os.environ.get("DATABASE_URL") or DEFAULT_URL
    kwargs: dict = {"echo": echo, "future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url or url == "sqlite://":
            from sqlalchemy.pool import StaticPool

            kwargs["poolclass"] = StaticPool
    engine = create_engine(url, **kwargs)
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn, _record):  # noqa: ANN001
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

    return engine


def create_schema(engine: Engine) -> None:
    """Tables plus immutability triggers (tests and the demo). Production uses `alembic upgrade head`, which runs the same trigger code."""
    Base.metadata.create_all(engine)
    install_triggers(engine)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    s = factory()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
