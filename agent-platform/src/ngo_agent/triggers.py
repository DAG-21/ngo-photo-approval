"""Database-level immutability. The rules live in the database so that no application bug (or script) can rewrite what the AI said.

Rules (docs/PHASE2A_DESIGN.md section 5):
  evaluation_runs       UPDATE refused once AI_EVALUATED or FAILED; DELETE refused
  criterion_corrections UPDATE refused (a new revision is a new row); DELETE refused
  feedback_events       UPDATE and DELETE refused
  lifecycle_events      UPDATE and DELETE refused
  photos                UPDATE refused when the stored file identity would change; DELETE refused
  human_reviews         UPDATE refused once FEEDBACK_SUBMITTED; DELETE refused
  photo_embeddings      UPDATE and DELETE refused (another model means another row)
  example_flags         UPDATE and DELETE refused (append-only)
  gap_analyses          UPDATE and DELETE refused (append-only)
  improvement_cycles    DELETE refused (the status of a cycle moves from RUNNING to a final one)
  change_records, change_events, change_tests   UPDATE and DELETE refused (append-only)
  note_drafts, note_events   UPDATE and DELETE refused (append-only)
  rule_edits, rule_edit_events   UPDATE and DELETE refused (append-only)
Works on SQLite and PostgreSQL. The same code runs in tests (create_all) and in the Alembic migration.
"""

from __future__ import annotations

from typing import Iterable, Optional, Union

from sqlalchemy import Connection, Engine

PHOTO_FIXED = ["image_storage_path", "file_hash", "organisation_id", "uploaded_at", "uploaded_by"]
# (trigger name, table, operation, condition on OLD/NEW or None for "always", message)
RULES: list[tuple[str, str, str, Union[str, list[str], None], str]] = [
    ("trg_eval_frozen", "evaluation_runs", "UPDATE", "OLD.processing_status IN ('AI_EVALUATED','FAILED')", "evaluation_runs are immutable once completed"),
    ("trg_eval_no_delete", "evaluation_runs", "DELETE", None, "evaluation_runs can never be deleted"),
    ("trg_corr_frozen", "criterion_corrections", "UPDATE", None, "a correction cannot be edited; record a new revision"),
    ("trg_corr_no_delete", "criterion_corrections", "DELETE", None, "corrections can never be deleted"),
    ("trg_fbev_no_update", "feedback_events", "UPDATE", None, "feedback_events are append-only"),
    ("trg_fbev_no_delete", "feedback_events", "DELETE", None, "feedback_events are append-only"),
    ("trg_lcev_no_update", "lifecycle_events", "UPDATE", None, "lifecycle_events are append-only"),
    ("trg_lcev_no_delete", "lifecycle_events", "DELETE", None, "lifecycle_events are append-only"),
    ("trg_photo_fixed", "photos", "UPDATE", PHOTO_FIXED, "the stored identity of a photo cannot change"),
    ("trg_photo_no_delete", "photos", "DELETE", None, "photos cannot be deleted here (use the retention workflow)"),
    ("trg_review_frozen", "human_reviews", "UPDATE", "OLD.review_status = 'FEEDBACK_SUBMITTED'", "a submitted review cannot be edited"),
    ("trg_review_no_delete", "human_reviews", "DELETE", None, "reviews can never be deleted"),
    ("trg_embed_no_update", "photo_embeddings", "UPDATE", None, "an embedding cannot be edited; another model makes another row"),
    ("trg_embed_no_delete", "photo_embeddings", "DELETE", None, "embeddings cannot be deleted here"),
    ("trg_flag_no_update", "example_flags", "UPDATE", None, "example_flags are append-only"),
    ("trg_flag_no_delete", "example_flags", "DELETE", None, "example_flags are append-only"),
    ("trg_gap_no_update", "gap_analyses", "UPDATE", None, "gap_analyses are append-only"),
    ("trg_gap_no_delete", "gap_analyses", "DELETE", None, "gap_analyses are append-only"),
    ("trg_cycle_no_delete", "improvement_cycles", "DELETE", None, "improvement cycles can never be deleted"),
    ("trg_chg_no_update", "change_records", "UPDATE", None, "change_records are append-only"),
    ("trg_chg_no_delete", "change_records", "DELETE", None, "change_records are append-only"),
    ("trg_chev_no_update", "change_events", "UPDATE", None, "change_events are append-only"),
    ("trg_chev_no_delete", "change_events", "DELETE", None, "change_events are append-only"),
    ("trg_chtest_no_update", "change_tests", "UPDATE", None, "change_tests are append-only"),
    ("trg_chtest_no_delete", "change_tests", "DELETE", None, "change_tests are append-only"),
    ("trg_note_no_update", "note_drafts", "UPDATE", None, "note_drafts are append-only"),
    ("trg_note_no_delete", "note_drafts", "DELETE", None, "note_drafts are append-only"),
    ("trg_noteev_no_update", "note_events", "UPDATE", None, "note_events are append-only"),
    ("trg_noteev_no_delete", "note_events", "DELETE", None, "note_events are append-only"),
    ("trg_ruleedit_no_update", "rule_edits", "UPDATE", None, "rule_edits are append-only"),
    ("trg_ruleedit_no_delete", "rule_edits", "DELETE", None, "rule_edits are append-only"),
    ("trg_ruleev_no_update", "rule_edit_events", "UPDATE", None, "rule_edit_events are append-only"),
    ("trg_ruleev_no_delete", "rule_edit_events", "DELETE", None, "rule_edit_events are append-only"),
]


def _when(cond: Union[str, list[str], None], dialect: str) -> str | None:
    if cond is None:
        return None
    if isinstance(cond, str):
        return cond
    op = "IS NOT" if dialect == "sqlite" else "IS DISTINCT FROM"
    return " OR ".join(f"OLD.{c} {op} NEW.{c}" for c in cond)


# The tables that exist at each migration revision. A migration installs and removes only the triggers of the tables it can see: the rule list above
# is the current one, and a table added later (photo_embeddings, example_flags: revision 0003) does not exist yet when an earlier revision runs.
BASE_TABLES = frozenset({"evaluation_runs", "criterion_corrections", "feedback_events", "lifecycle_events", "photos", "human_reviews"})   # revisions 0001 and 0002
TABLES_0003 = BASE_TABLES | {"photo_embeddings", "example_flags"}                                                                        # revision 0003
TABLES_0004 = TABLES_0003 | {"gap_analyses"}                                                                                              # revision 0004
TABLES_0005 = TABLES_0004 | {"improvement_cycles", "change_records", "change_events", "change_tests"}                                       # revision 0005
TABLES_0006 = TABLES_0005 | {"note_drafts", "note_events"}                                                                                 # revision 0006
ALL_TABLES = TABLES_0006 | {"rule_edits", "rule_edit_events"}                                                                              # revision 0007 and later


def _rules(tables: Optional[Iterable[str]]) -> list[tuple[str, str, str, Union[str, list[str], None], str]]:
    return [r for r in RULES if tables is None or r[1] in set(tables)]


def trigger_statements(dialect: str, tables: Optional[Iterable[str]] = None) -> list[str]:
    """The statements that install the immutability triggers (for `tables` only, or all of them)."""
    out: list[str] = []
    if dialect == "postgresql":
        out.append("CREATE OR REPLACE FUNCTION ngo_block() RETURNS trigger AS $$ BEGIN RAISE EXCEPTION USING MESSAGE = TG_ARGV[0], ERRCODE = 'integrity_constraint_violation'; END; $$ LANGUAGE plpgsql")
    for name, table, op, cond, msg in _rules(tables):
        when = _when(cond, dialect)
        if dialect == "sqlite":
            w = f" WHEN {when}" if when else ""
            out.append(f"CREATE TRIGGER IF NOT EXISTS {name} BEFORE {op} ON {table}{w} BEGIN SELECT RAISE(ABORT, '{msg}'); END")
        elif dialect == "postgresql":
            w = f" WHEN ({when})" if when else ""
            out.append(f"DROP TRIGGER IF EXISTS {name} ON {table}")
            out.append(f"CREATE TRIGGER {name} BEFORE {op} ON {table} FOR EACH ROW{w} EXECUTE FUNCTION ngo_block('{msg}')")
        else:
            raise NotImplementedError(f"immutability triggers are not written for dialect {dialect!r}")
    return out


def drop_statements(dialect: str, tables: Optional[Iterable[str]] = None) -> list[str]:
    out = [f"DROP TRIGGER IF EXISTS {name}" + (f" ON {table}" if dialect == "postgresql" else "") for name, table, *_ in _rules(tables)]
    if dialect == "postgresql":
        out.append("DROP FUNCTION IF EXISTS ngo_block()")
    return out


def _run(bind: Union[Engine, Connection], statements: list[str]) -> None:
    if isinstance(bind, Engine):
        with bind.begin() as conn:
            for s in statements:
                conn.exec_driver_sql(s)
    else:
        for s in statements:
            bind.exec_driver_sql(s)


def install_triggers(bind: Union[Engine, Connection], tables: Optional[Iterable[str]] = None) -> None:
    _run(bind, trigger_statements(bind.dialect.name, tables))


def remove_triggers(bind: Union[Engine, Connection], tables: Optional[Iterable[str]] = None) -> None:
    _run(bind, drop_statements(bind.dialect.name, tables))
