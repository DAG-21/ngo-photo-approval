"""learning loop (L4): cycles, drafted changes, their status history and their tests

Adds `improvement_cycles` (the only mutable row: its status), `change_records`, `change_events` and `change_tests` (append-only). Every table is tied to one
organisation by a composite key. Nothing is backfilled. The immutability triggers are rebuilt so the new tables are covered.

Revision ID: 0005
Revises: 0004
"""
from alembic import context, op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from ngo_agent.triggers import TABLES_0004, TABLES_0005, drop_statements, install_triggers, remove_triggers, trigger_statements

revision = '0005'
down_revision = '0004'
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
    _triggers_off(TABLES_0004)   # the new tables do not exist yet
    op.create_table('improvement_cycles',
    sa.Column('cycle_id', sa.String(length=36), nullable=False),
    sa.Column('organisation_id', sa.String(length=36), nullable=False),
    sa.Column('ordinal', sa.Integer(), nullable=False),
    sa.Column('decided_count', sa.Integer(), nullable=False),
    sa.Column('trigger', sa.String(length=10), nullable=False),
    sa.Column('status', sa.String(length=30), nullable=False),
    sa.Column('summary', JSON, nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['organisation_id'], ['organisations.organisation_id'], ),
    sa.PrimaryKeyConstraint('cycle_id'),
    sa.UniqueConstraint('cycle_id', 'organisation_id', name='uq_cycle_org'),
    sa.UniqueConstraint('organisation_id', 'ordinal', name='uq_cycle_org_ordinal')
    )
    _index('improvement_cycles', 'organisation_id')

    op.create_table('change_records',
    sa.Column('change_id', sa.String(length=36), nullable=False),
    sa.Column('organisation_id', sa.String(length=36), nullable=False),
    sa.Column('cycle_id', sa.String(length=36), nullable=False),
    sa.Column('label', sa.String(length=20), nullable=False),
    sa.Column('fingerprint', sa.String(length=64), nullable=False),
    sa.Column('kind', sa.String(length=30), nullable=False),
    sa.Column('target', sa.String(length=40), nullable=True),
    sa.Column('loosening', sa.Boolean(), nullable=False),
    sa.Column('record', JSON, nullable=False),
    sa.Column('rationale', sa.Text(), nullable=False),
    sa.Column('evidence', JSON, nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['cycle_id', 'organisation_id'], ['improvement_cycles.cycle_id', 'improvement_cycles.organisation_id'], ),
    sa.ForeignKeyConstraint(['organisation_id'], ['organisations.organisation_id'], ),
    sa.PrimaryKeyConstraint('change_id'),
    sa.UniqueConstraint('change_id', 'organisation_id', name='uq_change_org'),
    sa.UniqueConstraint('organisation_id', 'fingerprint', name='uq_change_org_fingerprint'),
    sa.UniqueConstraint('organisation_id', 'label', name='uq_change_org_label')
    )
    _index('change_records', 'fingerprint', 'organisation_id')

    op.create_table('change_events',
    sa.Column('event_id', sa.String(length=36), nullable=False),
    sa.Column('organisation_id', sa.String(length=36), nullable=False),
    sa.Column('change_id', sa.String(length=36), nullable=False),
    sa.Column('seq', sa.Integer(), nullable=False),
    sa.Column('status', sa.String(length=24), nullable=False),
    sa.Column('reason', sa.Text(), nullable=False),
    sa.Column('payload', JSON, nullable=False),
    sa.Column('rulebook_version_id', sa.String(length=36), nullable=True),
    sa.Column('created_by', sa.String(length=36), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['change_id', 'organisation_id'], ['change_records.change_id', 'change_records.organisation_id'], ),
    sa.ForeignKeyConstraint(['created_by'], ['users.user_id'], ),
    sa.ForeignKeyConstraint(['organisation_id'], ['organisations.organisation_id'], ),
    sa.PrimaryKeyConstraint('event_id'),
    sa.UniqueConstraint('change_id', 'seq', name='uq_change_event_seq')
    )
    _index('change_events', 'change_id', 'organisation_id')

    op.create_table('change_tests',
    sa.Column('test_id', sa.String(length=36), nullable=False),
    sa.Column('organisation_id', sa.String(length=36), nullable=False),
    sa.Column('change_id', sa.String(length=36), nullable=False),
    sa.Column('cycle_id', sa.String(length=36), nullable=False),
    sa.Column('method', sa.String(length=20), nullable=False),
    sa.Column('metrics', JSON, nullable=False),
    sa.Column('flips', JSON, nullable=False),
    sa.Column('guardrails', JSON, nullable=False),
    sa.Column('passed', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['change_id', 'organisation_id'], ['change_records.change_id', 'change_records.organisation_id'], ),
    sa.ForeignKeyConstraint(['organisation_id'], ['organisations.organisation_id'], ),
    sa.PrimaryKeyConstraint('test_id')
    )
    _index('change_tests', 'change_id', 'organisation_id')
    _triggers_on(TABLES_0005)


def downgrade() -> None:
    _triggers_off(TABLES_0005)
    for table, cols in (('change_tests', ('organisation_id', 'change_id')), ('change_events', ('organisation_id', 'change_id')),
                        ('change_records', ('organisation_id', 'fingerprint')), ('improvement_cycles', ('organisation_id',))):
        with op.batch_alter_table(table, schema=None) as batch_op:
            for c in cols:
                batch_op.drop_index(batch_op.f(f'ix_{table}_{c}'))
        op.drop_table(table)
    _triggers_on(TABLES_0004)
