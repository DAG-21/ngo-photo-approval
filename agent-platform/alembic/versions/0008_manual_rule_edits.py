"""rule rewrites entered by an administrator: the link to a decision becomes optional

A rewrite an administrator enters by hand is not the outcome of one reviewer decision, so `rule_edits.review_id`, `evaluation_id` and `photo_id` may now be empty (the composite key to
`human_reviews` simply does not apply to such a row). Nothing is backfilled. The immutability triggers are taken off while the table is rebuilt and put back after.

Revision ID: 0008
Revises: 0007
"""
from alembic import context, op
import sqlalchemy as sa

from ngo_agent.triggers import ALL_TABLES, drop_statements, install_triggers, remove_triggers, trigger_statements

revision = '0008'
down_revision = '0007'
branch_labels = None
depends_on = None

LINKS = ('review_id', 'evaluation_id', 'photo_id')


def _triggers_off() -> None:
    if context.is_offline_mode():
        for stmt in drop_statements(op.get_context().dialect.name, ALL_TABLES):
            op.execute(stmt)
    else:
        remove_triggers(op.get_bind(), ALL_TABLES)


def _triggers_on() -> None:
    if context.is_offline_mode():
        for stmt in trigger_statements(op.get_context().dialect.name, ALL_TABLES):
            op.execute(stmt)
    else:
        install_triggers(op.get_bind(), ALL_TABLES)


def _links(nullable: bool) -> None:
    if op.get_context().dialect.name == 'sqlite':
        op.execute('PRAGMA defer_foreign_keys=ON')   # the table is rebuilt: its rows are checked again when the transaction ends, not while the old copy is dropped
    with op.batch_alter_table('rule_edits', schema=None) as batch_op:
        for c in LINKS:
            batch_op.alter_column(c, existing_type=sa.String(length=36), nullable=nullable)


def upgrade() -> None:
    _triggers_off()
    _links(True)
    _triggers_on()


def downgrade() -> None:
    _triggers_off()
    op.execute('DELETE FROM rule_edit_events WHERE edit_id IN (SELECT edit_id FROM rule_edits WHERE review_id IS NULL)')   # a hand-entered rewrite has no decision to point at
    op.execute('DELETE FROM rule_edits WHERE review_id IS NULL')
    _links(False)
    _triggers_on()
