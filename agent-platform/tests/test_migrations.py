"""Every migration can be applied and reversed one step at a time, on SQLite and (with TEST_DATABASE_URL) on PostgreSQL, and each revision has the right triggers.

A migration must install only the immutability triggers of the tables that exist at its own revision: the shared rule list describes the current schema,
and a table added later does not exist yet when an earlier revision runs.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
STEPS = ["0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008"]
TRIGGERS = {"0001": 12, "0002": 12, "0003": 16, "0004": 18, "0005": 25, "0006": 29, "0007": 33, "0008": 33}
TABLES_ADDED_IN_0003 = {"photo_embeddings", "example_flags"}
TABLES_ADDED_IN_0004 = {"gap_analyses"}
TABLES_ADDED_IN_0005 = {"improvement_cycles", "change_records", "change_events", "change_tests"}
TABLES_ADDED_IN_0006 = {"note_drafts", "note_events"}
TABLES_ADDED_IN_0007 = {"rule_edits", "rule_edit_events"}
NEW_TABLES = {"0001": set(), "0002": set(), "0003": TABLES_ADDED_IN_0003, "0004": TABLES_ADDED_IN_0003 | TABLES_ADDED_IN_0004,
              "0005": TABLES_ADDED_IN_0003 | TABLES_ADDED_IN_0004 | TABLES_ADDED_IN_0005,
              "0006": TABLES_ADDED_IN_0003 | TABLES_ADDED_IN_0004 | TABLES_ADDED_IN_0005 | TABLES_ADDED_IN_0006,
              "0007": TABLES_ADDED_IN_0003 | TABLES_ADDED_IN_0004 | TABLES_ADDED_IN_0005 | TABLES_ADDED_IN_0006 | TABLES_ADDED_IN_0007,
              "0008": TABLES_ADDED_IN_0003 | TABLES_ADDED_IN_0004 | TABLES_ADDED_IN_0005 | TABLES_ADDED_IN_0006 | TABLES_ADDED_IN_0007}   # 0008 adds no table, it loosens three columns   # what exists beyond the first eleven, per revision


def _config() -> Config:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    return cfg


@pytest.fixture()
def database(tmp_path):
    """(url, engine) of a fresh, empty database: a SQLite file, or a throwaway schema on the PostgreSQL server named by TEST_DATABASE_URL."""
    pg = os.environ.get("TEST_DATABASE_URL")
    if pg:
        schema = "m_" + uuid.uuid4().hex[:12]
        admin = create_engine(pg)
        with admin.begin() as c:
            c.exec_driver_sql(f"CREATE SCHEMA {schema}")
        url = make_url(pg).update_query_dict({"options": f"-csearch_path={schema}"}).render_as_string(hide_password=False)
        engine = create_engine(url)
        yield url, engine
        engine.dispose()
        with admin.begin() as c:
            c.exec_driver_sql(f"DROP SCHEMA {schema} CASCADE")
        admin.dispose()
    else:
        url = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
        engine = create_engine(url)
        yield url, engine
        engine.dispose()


def _trigger_count(engine) -> int:
    with engine.connect() as c:
        if engine.dialect.name == "sqlite":
            return c.execute(text("SELECT count(*) FROM sqlite_master WHERE type='trigger'")).scalar()
        return c.execute(text("SELECT count(*) FROM pg_trigger t JOIN pg_class r ON r.oid=t.tgrelid JOIN pg_namespace n ON n.oid=r.relnamespace "
                              "WHERE NOT t.tgisinternal AND n.nspname=current_schema()")).scalar()


def _tables(engine) -> set[str]:
    return set(inspect(engine).get_table_names()) - {"alembic_version"}


def test_every_migration_goes_up_and_down_one_step_at_a_time_with_the_right_triggers(database, monkeypatch):
    url, engine = database
    monkeypatch.setenv("DATABASE_URL", url)
    cfg = _config()
    everything_new = NEW_TABLES["0008"]
    for rev in STEPS:                                           # up, one revision at a time
        command.upgrade(cfg, rev)
        assert _trigger_count(engine) == TRIGGERS[rev], rev
        assert (everything_new & _tables(engine)) == NEW_TABLES[rev], rev
    for rev in reversed(STEPS[:-1]):                            # and back down, one revision at a time
        command.downgrade(cfg, rev)
        assert _trigger_count(engine) == TRIGGERS[rev], rev
        assert (everything_new & _tables(engine)) == NEW_TABLES[rev], rev
    command.downgrade(cfg, "base")
    assert _tables(engine) == set() and _trigger_count(engine) == 0
    command.upgrade(cfg, "head")                                # and a fresh upgrade straight to the head
    assert _trigger_count(engine) == TRIGGERS["0008"] and everything_new <= _tables(engine)
    command.check(cfg)                                          # the models and the migrations describe the same schema


def test_after_0008_the_link_columns_are_optional_and_every_constraint_and_index_survived_the_rebuild(database, monkeypatch):
    url, engine = database
    monkeypatch.setenv("DATABASE_URL", url)
    command.upgrade(_config(), "head")
    insp = inspect(engine)
    nullable = {c["name"]: c["nullable"] for c in insp.get_columns("rule_edits")}
    assert nullable["review_id"] and nullable["evaluation_id"] and nullable["photo_id"] and not nullable["target_id"] and not nullable["organisation_id"] and not nullable["new_text"]
    assert {"uq_ruleedit_org", "uq_ruleedit_review_target"} <= {u["name"] for u in insp.get_unique_constraints("rule_edits")}
    assert {"organisations", "human_reviews"} <= {f["referred_table"] for f in insp.get_foreign_keys("rule_edits")}
    assert {"ix_rule_edits_organisation_id", "ix_rule_edits_review_id"} <= {i["name"] for i in insp.get_indexes("rule_edits")}
    assert "rule_edits" in {f["referred_table"] for f in insp.get_foreign_keys("rule_edit_events")} and {"uq_ruleedit_event_seq"} <= {u["name"] for u in insp.get_unique_constraints("rule_edit_events")}
