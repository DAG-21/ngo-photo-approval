"""example store (L2): image embeddings, example flags, and the examples shown to the model on an evaluation run

Adds `photo_embeddings` and `example_flags` (both immutable, with composite foreign keys that tie them to one organisation), and a nullable
`evaluation_runs.few_shot` column. Nothing is backfilled: existing runs have no examples, and photographs are embedded when uploaded.
The immutability triggers are rebuilt so the two new tables are covered.

Revision ID: 0003
Revises: 0002
"""
from alembic import context, op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from ngo_agent.triggers import BASE_TABLES, TABLES_0003, drop_statements, install_triggers, remove_triggers, trigger_statements

revision = '0003'
down_revision = '0002'
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
    _triggers_off(BASE_TABLES)   # the two new tables do not exist yet
    op.create_table('photo_embeddings',
    sa.Column('embedding_id', sa.String(length=36), nullable=False),
    sa.Column('photo_id', sa.String(length=36), nullable=False),
    sa.Column('organisation_id', sa.String(length=36), nullable=False),
    sa.Column('model', sa.String(length=80), nullable=False),
    sa.Column('dim', sa.Integer(), nullable=False),
    sa.Column('vector', JSON, nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['organisation_id'], ['organisations.organisation_id'], ),
    sa.ForeignKeyConstraint(['photo_id', 'organisation_id'], ['photos.photo_id', 'photos.organisation_id'], ),
    sa.PrimaryKeyConstraint('embedding_id'),
    sa.UniqueConstraint('photo_id', 'model', name='uq_embedding_photo_model')
    )
    with op.batch_alter_table('photo_embeddings', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_photo_embeddings_organisation_id'), ['organisation_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_photo_embeddings_photo_id'), ['photo_id'], unique=False)

    op.create_table('example_flags',
    sa.Column('flag_id', sa.String(length=36), nullable=False),
    sa.Column('photo_id', sa.String(length=36), nullable=False),
    sa.Column('organisation_id', sa.String(length=36), nullable=False),
    sa.Column('action', sa.String(length=10), nullable=False),
    sa.Column('reason', sa.Text(), nullable=False),
    sa.Column('created_by', sa.String(length=36), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['created_by'], ['users.user_id'], ),
    sa.ForeignKeyConstraint(['organisation_id'], ['organisations.organisation_id'], ),
    sa.ForeignKeyConstraint(['photo_id', 'organisation_id'], ['photos.photo_id', 'photos.organisation_id'], ),
    sa.PrimaryKeyConstraint('flag_id')
    )
    with op.batch_alter_table('example_flags', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_example_flags_organisation_id'), ['organisation_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_example_flags_photo_id'), ['photo_id'], unique=False)

    op.add_column('evaluation_runs', sa.Column('few_shot', JSON, nullable=True))
    _triggers_on(TABLES_0003)


def downgrade() -> None:
    _triggers_off(TABLES_0003)
    with op.batch_alter_table('evaluation_runs', schema=None) as batch_op:
        batch_op.drop_column('few_shot')
    with op.batch_alter_table('example_flags', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_example_flags_photo_id'))
        batch_op.drop_index(batch_op.f('ix_example_flags_organisation_id'))
    op.drop_table('example_flags')
    with op.batch_alter_table('photo_embeddings', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_photo_embeddings_photo_id'))
        batch_op.drop_index(batch_op.f('ix_photo_embeddings_organisation_id'))
    op.drop_table('photo_embeddings')
    _triggers_on(BASE_TABLES)
