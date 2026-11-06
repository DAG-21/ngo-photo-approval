"""remove the validation workflow

A person's submitted review is final: there is no validator role and no per-correction validation. Corrections become fully immutable
(a changed mind is a new revision), and learning examples no longer carry a validation status.

Upgrade: VALIDATOR users become REVIEWER; the validation columns are dropped (their history is not kept); the immutability triggers are rebuilt.
Downgrade restores the columns with their defaults (validation history is not restored) and installs the current triggers, in which a correction
row is fully frozen.

Revision ID: 0002
Revises: 0001
"""
from alembic import context, op
import sqlalchemy as sa

from ngo_agent.triggers import BASE_TABLES, drop_statements, install_triggers, remove_triggers, trigger_statements

revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def _triggers_off() -> None:
    if context.is_offline_mode():
        for stmt in drop_statements(op.get_context().dialect.name, BASE_TABLES):   # only the tables that exist at this revision
            op.execute(stmt)
    else:
        remove_triggers(op.get_bind(), BASE_TABLES)


def _triggers_on() -> None:
    if context.is_offline_mode():
        for stmt in trigger_statements(op.get_context().dialect.name, BASE_TABLES):
            op.execute(stmt)
    else:
        install_triggers(op.get_bind(), BASE_TABLES)


def upgrade() -> None:
    _triggers_off()   # also needed because SQLite rebuilds a table to drop a column, and a rebuilt table loses its triggers
    op.execute("UPDATE users SET role = 'REVIEWER' WHERE role = 'VALIDATOR'")
    with op.batch_alter_table('criterion_corrections', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_criterion_corrections_validation_status'))
        batch_op.drop_column('validation_comment')
        batch_op.drop_column('validated_at')
        batch_op.drop_column('validated_by')
        batch_op.drop_column('validation_status')
    with op.batch_alter_table('learning_examples', schema=None) as batch_op:
        batch_op.drop_column('validation_status')
    _triggers_on()


def downgrade() -> None:
    _triggers_off()
    with op.batch_alter_table('learning_examples', schema=None) as batch_op:
        batch_op.add_column(sa.Column('validation_status', sa.String(length=30), nullable=False, server_default='APPROVED_FOR_LEARNING'))
    with op.batch_alter_table('criterion_corrections', schema=None) as batch_op:
        batch_op.add_column(sa.Column('validation_status', sa.String(length=30), nullable=False, server_default='PENDING_VALIDATION'))
        batch_op.add_column(sa.Column('validated_by', sa.String(length=36), sa.ForeignKey('users.user_id', name='fk_corrections_validated_by'), nullable=True))   # named: SQLite's batch mode needs it
        batch_op.add_column(sa.Column('validated_at', sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column('validation_comment', sa.Text(), nullable=True))
        batch_op.create_index(batch_op.f('ix_criterion_corrections_validation_status'), ['validation_status'], unique=False)
    _triggers_on()
