"""gap analysis (L3): the gaps between what the AI said and what the person wrote, and a mark on examples whose reason is too generic to teach anything

Adds `gap_analyses` (append-only, tied to one organisation and one review) and `learning_examples.low_information`. Existing examples are not
re-judged: they keep `low_information = false`. The immutability triggers are rebuilt so the new table is covered.

Revision ID: 0004
Revises: 0003
"""
from alembic import context, op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from ngo_agent.triggers import TABLES_0003, TABLES_0004, drop_statements, install_triggers, remove_triggers, trigger_statements

revision = '0004'
down_revision = '0003'
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


def upgrade() -> None:
    _triggers_off(TABLES_0003)   # the new table does not exist yet
    op.create_table('gap_analyses',
    sa.Column('gap_id', sa.String(length=36), nullable=False),
    sa.Column('organisation_id', sa.String(length=36), nullable=False),
    sa.Column('review_id', sa.String(length=36), nullable=False),
    sa.Column('evaluation_id', sa.String(length=36), nullable=False),
    sa.Column('photo_id', sa.String(length=36), nullable=False),
    sa.Column('analyzer', sa.String(length=60), nullable=False),
    sa.Column('model_id', sa.String(length=100), nullable=True),
    sa.Column('status', sa.String(length=10), nullable=False),
    sa.Column('agreement', sa.String(length=20), nullable=False),
    sa.Column('low_information', sa.Boolean(), nullable=False),
    sa.Column('gaps', JSON, nullable=False),
    sa.Column('summary', sa.Text(), nullable=False),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('cost_usd', sa.Float(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['organisation_id'], ['organisations.organisation_id'], ),
    sa.ForeignKeyConstraint(['review_id', 'organisation_id'], ['human_reviews.review_id', 'human_reviews.organisation_id'], ),
    sa.PrimaryKeyConstraint('gap_id'),
    sa.UniqueConstraint('review_id', 'analyzer', name='uq_gap_review_analyzer')
    )
    with op.batch_alter_table('gap_analyses', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_gap_analyses_organisation_id'), ['organisation_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_gap_analyses_photo_id'), ['photo_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_gap_analyses_review_id'), ['review_id'], unique=False)

    op.add_column('learning_examples', sa.Column('low_information', sa.Boolean(), nullable=False, server_default=sa.false()))
    _triggers_on(TABLES_0004)


def downgrade() -> None:
    _triggers_off(TABLES_0004)
    with op.batch_alter_table('learning_examples', schema=None) as batch_op:
        batch_op.drop_column('low_information')
    with op.batch_alter_table('gap_analyses', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_gap_analyses_review_id'))
        batch_op.drop_index(batch_op.f('ix_gap_analyses_photo_id'))
        batch_op.drop_index(batch_op.f('ix_gap_analyses_organisation_id'))
    op.drop_table('gap_analyses')
    _triggers_on(TABLES_0003)
