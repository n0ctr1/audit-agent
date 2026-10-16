"""initial_schema

Revision ID: 065ae34a4ccc
Revises:
Create Date: 2026-04-16

"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "065ae34a4ccc"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "audits",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("mode", sa.String(20), nullable=False),
        sa.Column("platform", sa.String(20)),
        sa.Column("target_dir", sa.Text),
        sa.Column("status", sa.String(20), nullable=False, server_default="running"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("budget_tokens", sa.Integer),
        sa.Column("budget_usd", sa.Numeric(10, 2)),
        sa.Column("tokens_used", sa.Integer, server_default="0"),
        sa.Column("cost_usd", sa.Numeric(10, 2), server_default="0"),
        sa.Column("metadata", postgresql.JSONB),
    )

    op.create_table(
        "findings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("audit_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("audits.id", ondelete="CASCADE")),
        sa.Column("hypothesis_id", postgresql.UUID(as_uuid=True)),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("vulnerability_class", sa.String(50)),
        sa.Column("severity", sa.String(20), nullable=False),
        sa.Column("confidence", sa.Numeric(3, 2)),
        sa.Column("status", sa.String(20), nullable=False, server_default="unverified"),
        sa.Column("affected_functions", postgresql.JSONB),
        sa.Column("description", sa.Text),
        sa.Column("poc_uri", sa.Text),
        sa.Column("trace_uri", sa.Text),
        sa.Column("report_uri", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("validated_at", sa.DateTime(timezone=True)),
        sa.Column("human_review", postgresql.JSONB),
    )

    op.create_table(
        "hypotheses",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("audit_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("audits.id", ondelete="CASCADE")),
        sa.Column("title", sa.Text),
        sa.Column("vulnerability_class", sa.String(50)),
        sa.Column("priority_score", sa.Numeric(5, 2)),
        sa.Column("attack_scenario", sa.Text),
        sa.Column("preconditions", sa.Text),
        sa.Column("similar_cases", postgresql.JSONB),
        sa.Column("subgraph_context", postgresql.JSONB),
        sa.Column("status", sa.String(20), server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "tool_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("audit_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("audits.id", ondelete="CASCADE")),
        sa.Column("tool_name", sa.String(50), nullable=False),
        sa.Column("command", sa.Text),
        sa.Column("status", sa.String(20)),
        sa.Column("exit_code", sa.Integer),
        sa.Column("stdout_uri", sa.Text),
        sa.Column("stderr_uri", sa.Text),
        sa.Column("duration_ms", sa.Integer),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("ended_at", sa.DateTime(timezone=True)),
    )

    op.create_table(
        "debate_rounds",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("finding_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("findings.id", ondelete="CASCADE")),
        sa.Column("round_number", sa.Integer),
        sa.Column("hunter_argument", sa.Text),
        sa.Column("skeptic_argument", sa.Text),
        sa.Column("resolution", sa.Text),
        sa.Column("adjudication_poc_uri", sa.Text),
    )

    op.create_table(
        "feedback_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("audit_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("audits.id")),
        sa.Column("finding_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("findings.id")),
        sa.Column("event_type", sa.String(30)),
        sa.Column("ground_truth", postgresql.JSONB),
        sa.Column("reviewer", sa.Text),
        sa.Column("notes", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # LangGraph checkpoints table is intentionally NOT created here.
    # AsyncPostgresSaver.setup() manages its own schema with the correct columns.
    # Creating it here with an older schema causes column-mismatch errors.


def downgrade() -> None:
    op.drop_table("feedback_events")
    op.drop_table("debate_rounds")
    op.drop_table("tool_runs")
    op.drop_table("hypotheses")
    op.drop_table("findings")
    op.drop_table("audits")
