"""prompt notes (L5): drafted notes and their status history

Adds `note_drafts` (what the drafter proposed for one gate after one decision, and how the checks judged it) and `note_events` (the append-only status history:
DRAFTED, REJECTED, HELD, APPLIED, CONFIRMED, SUPERSEDED, WITHDRAWN). Both are append-only and tied to one organisation by a composite key. Nothing is backfilled.
The immutability triggers are rebuilt so the new tables are covered.

Revision ID: 0006
Revises: 0005
"""
from alembic import context, op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from ngo_agent.triggers import TABLES_0005, TABLES_0006, drop_statements, install_triggers, remove_triggers, trigger_statements

revision = '0006'
down_revision = '0005'
branch_labels = None
depends_on = None

JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql')


def _triggers_off(tables) -> None:  # noqa: ANN001
    if context.is_offline_mode():
        for stmt in drop_statements(op.get_context().dialect.name, tables):
            op.execute(stmt)
    else:
        remove_triggers(op.get_bind(), tables)


def _triggers_on(tables) -> None:  # noqa: ANN001
    if context.is_offline_mode():
        for stmt in trigger_statements(op.get_context().dialect.name, tables):
            op.execute(stmt)
    else:
        install_triggers(op.get_bind(), tables)


def _index(table: str, *columns: str) -> None:
    with op.batch_alter_table(table, schema=None) as batch_op:
        for c in columns:
            batch_op.create_index(batch_op.f(f'ix_{table}_{c}'), [c], unique=False)


def upgrade() -> None:
    _triggers_off(TABLES_0005)   # the new tables do not exist yet
    op.create_table('note_drafts',
    sa.Column('draft_id', sa.String(length=36), nullable=False),
    sa.Column('organisation_id', sa.String(length=36), nullable=False),
    sa.Column('review_id', sa.String(length=36), nullable=False),
    sa.Column('evaluation_id', sa.String(length=36), nullable=False),
    sa.Column('photo_id', sa.String(length=36), nullable=False),
    sa.Column('gate_id', sa.String(length=10), nullable=False),
    sa.Column('outcome', sa.String(length=24), nullable=False),
    sa.Column('direction', sa.String(length=8), nullable=False),
    sa.Column('note_text', sa.Text(), nullable=False),
    sa.Column('rationale', sa.Text(), nullable=False),
    sa.Column('contradicts_client_wording', sa.Boolean(), nullable=False),
    sa.Column('client_gate', sa.Boolean(), nullable=False),
    sa.Column('flags', JSON, nullable=False),
    sa.Column('checks', JSON, nullable=False),
    sa.Column('source', JSON, nullable=False),
    sa.Column('model_id', sa.String(length=100), nullable=True),
    sa.Column('cost_usd', sa.Float(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['organisation_id'], ['organisations.organisation_id'], ),
    sa.ForeignKeyConstraint(['review_id', 'organisation_id'], ['human_reviews.review_id', 'human_reviews.organisation_id'], ),
    sa.PrimaryKeyConstraint('draft_id'),
    sa.UniqueConstraint('draft_id', 'organisation_id', name='uq_note_org'),
    sa.UniqueConstraint('review_id', 'gate_id', name='uq_note_review_gate')
    )
    _index('note_drafts', 'organisation_id', 'review_id')

    op.create_table('note_events',
    sa.Column('event_id', sa.String(length=36), nullable=False),
    sa.Column('organisation_id', sa.String(length=36), nullable=False),
    sa.Column('draft_id', sa.String(length=36), nullable=False),
    sa.Column('seq', sa.Integer(), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('reason', sa.Text(), nullable=False),
    sa.Column('payload', JSON, nullable=False),
    sa.Column('prompt_version_id', sa.String(length=36), nullable=True),
    sa.Column('created_by', sa.String(length=36), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['created_by'], ['users.user_id'], ),
    sa.ForeignKeyConstraint(['draft_id', 'organisation_id'], ['note_drafts.draft_id', 'note_drafts.organisation_id'], ),
    sa.ForeignKeyConstraint(['organisation_id'], ['organisations.organisation_id'], ),
    sa.PrimaryKeyConstraint('event_id'),
    sa.UniqueConstraint('draft_id', 'seq', name='uq_note_event_seq')
    )
    _index('note_events', 'organisation_id', 'draft_id')
    _triggers_on(TABLES_0006)


def downgrade() -> None:
    _triggers_off(TABLES_0006)
    for table, cols in (('note_events', ('organisation_id', 'draft_id')), ('note_drafts', ('organisation_id', 'review_id'))):
        with op.batch_alter_table(table, schema=None) as batch_op:
            for c in cols:
                batch_op.drop_index(batch_op.f(f'ix_{table}_{c}'))
        op.drop_table(table)
    _triggers_on(TABLES_0005)
