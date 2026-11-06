"""Alembic environment. Run: DATABASE_URL=postgresql+psycopg://... alembic upgrade head"""
import os
import sys
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ngo_agent.models import Base  # noqa: E402

target_metadata = Base.metadata


def _url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("set DATABASE_URL (for example sqlite:///./ngo_agent.db or postgresql+psycopg://user:pass@host/db)")
    return url


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_url())
    with engine.connect() as conn:
        context.configure(connection=conn, target_metadata=target_metadata, render_as_batch=conn.dialect.name == "sqlite")
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
