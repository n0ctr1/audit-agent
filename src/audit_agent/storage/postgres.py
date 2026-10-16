"""SQLAlchemy ORM models and async engine."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import (
    NUMERIC,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class AuditRun(Base):
    __tablename__ = "audits"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    mode: Mapped[str] = mapped_column(String(20), nullable=False)
    platform: Mapped[str | None] = mapped_column(String(20))
    target_dir: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="running")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    budget_tokens: Mapped[int | None] = mapped_column(Integer)
    budget_usd: Mapped[float | None] = mapped_column(NUMERIC(10, 2))
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(NUMERIC(10, 2), default=0)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB)

    findings: Mapped[list[Finding]] = relationship("Finding", back_populates="audit", cascade="all, delete-orphan")
    hypotheses: Mapped[list[HypothesisRecord]] = relationship("HypothesisRecord", back_populates="audit", cascade="all, delete-orphan")
    tool_runs: Mapped[list[ToolRun]] = relationship("ToolRun", back_populates="audit", cascade="all, delete-orphan")


class Finding(Base):
    __tablename__ = "findings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    audit_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("audits.id", ondelete="CASCADE"))
    hypothesis_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    title: Mapped[str] = mapped_column(Text, nullable=False)
    vulnerability_class: Mapped[str | None] = mapped_column(String(50))
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[float | None] = mapped_column(NUMERIC(3, 2))
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="unverified")
    affected_functions: Mapped[list | None] = mapped_column(JSONB)
    description: Mapped[str | None] = mapped_column(Text)
    poc_uri: Mapped[str | None] = mapped_column(Text)
    trace_uri: Mapped[str | None] = mapped_column(Text)
    report_uri: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    validated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    human_review: Mapped[dict | None] = mapped_column(JSONB)

    audit: Mapped[AuditRun] = relationship("AuditRun", back_populates="findings")


class HypothesisRecord(Base):
    __tablename__ = "hypotheses"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    audit_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("audits.id", ondelete="CASCADE"))
    title: Mapped[str | None] = mapped_column(Text)
    vulnerability_class: Mapped[str | None] = mapped_column(String(50))
    priority_score: Mapped[float | None] = mapped_column(NUMERIC(5, 2))
    attack_scenario: Mapped[str | None] = mapped_column(Text)
    preconditions: Mapped[str | None] = mapped_column(Text)
    similar_cases: Mapped[list | None] = mapped_column(JSONB)
    subgraph_context: Mapped[dict | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    audit: Mapped[AuditRun] = relationship("AuditRun", back_populates="hypotheses")


class ToolRun(Base):
    __tablename__ = "tool_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    audit_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("audits.id", ondelete="CASCADE"))
    tool_name: Mapped[str] = mapped_column(String(50), nullable=False)
    command: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str | None] = mapped_column(String(20))
    exit_code: Mapped[int | None] = mapped_column(Integer)
    stdout_uri: Mapped[str | None] = mapped_column(Text)
    stderr_uri: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    audit: Mapped[AuditRun] = relationship("AuditRun", back_populates="tool_runs")


class DebateRound(Base):
    __tablename__ = "debate_rounds"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    finding_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("findings.id", ondelete="CASCADE"))
    round_number: Mapped[int] = mapped_column(Integer)
    hunter_argument: Mapped[str | None] = mapped_column(Text)
    skeptic_argument: Mapped[str | None] = mapped_column(Text)
    resolution: Mapped[str | None] = mapped_column(Text)
    adjudication_poc_uri: Mapped[str | None] = mapped_column(Text)


class FeedbackEvent(Base):
    __tablename__ = "feedback_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    audit_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("audits.id"))
    finding_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("findings.id"))
    event_type: Mapped[str | None] = mapped_column(String(30))
    ground_truth: Mapped[dict | None] = mapped_column(JSONB)
    reviewer: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ConfirmedPattern(Base):
    __tablename__ = "confirmed_patterns"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    finding_id: Mapped[str | None] = mapped_column(Text)
    audit_id: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text)
    vulnerability_class: Mapped[str | None] = mapped_column(String(80))
    affected_functions: Mapped[list | None] = mapped_column(JSONB)
    description: Mapped[str | None] = mapped_column(Text)
    severity: Mapped[str | None] = mapped_column(String(20))
    confirmed_at: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class FpSuppressionSet(Base):
    __tablename__ = "fp_suppression_set"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    finding_id: Mapped[str | None] = mapped_column(Text)
    audit_id: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text)
    vulnerability_class: Mapped[str | None] = mapped_column(String(80))
    notes: Mapped[str | None] = mapped_column(Text)
    added_at: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    finding_id: Mapped[str | None] = mapped_column(Text)
    audit_id: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text)
    vulnerability_class: Mapped[str | None] = mapped_column(String(80))
    affected_functions: Mapped[list | None] = mapped_column(JSONB)
    description: Mapped[str | None] = mapped_column(Text)
    severity: Mapped[str | None] = mapped_column(String(20))
    entry_type: Mapped[str | None] = mapped_column(String(40))
    added_at: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# LangGraph checkpoints table is managed by langgraph's own saver — we just declare it for Alembic
class Checkpoint(Base):
    __tablename__ = "checkpoints"

    thread_id: Mapped[str] = mapped_column(Text, primary_key=True)
    checkpoint_id: Mapped[str] = mapped_column(Text, primary_key=True)
    parent_id: Mapped[str | None] = mapped_column(Text)
    type: Mapped[str | None] = mapped_column(Text)
    checkpoint: Mapped[dict | None] = mapped_column(JSONB)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ── Engine / session factory ──────────────────────────────────────────────────

_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine(url: str | None = None):  # type: ignore[return]
    global _engine
    if _engine is None:
        from audit_agent.config import get_settings

        _engine = create_async_engine(url or get_settings().postgres_url, echo=False, pool_pre_ping=True)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with get_session_factory()() as session:
        yield session


def reset_engine() -> None:
    """Discard cached engine + session factory. Useful in tests."""
    global _engine, _session_factory
    _engine = None
    _session_factory = None
