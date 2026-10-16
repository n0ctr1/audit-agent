"""Integration test fixtures — requires Docker (testcontainers)."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def postgres_container():
    """Start a real Postgres container for the test session (skipped without Docker)."""
    try:
        import docker

        docker.from_env().ping()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Docker is not available: {exc}")

    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


@pytest.fixture()
def pg_urls(postgres_container):
    """Return (async_url, sync_url) for the running Postgres container."""
    # testcontainers returns a psycopg2 URL; rewrite drivers as needed
    raw = postgres_container.get_connection_url()
    sync_url = raw.replace("+psycopg2", "")          # plain postgresql://
    async_url = raw.replace("+psycopg2", "+asyncpg") # for SQLAlchemy async
    return async_url, sync_url


@pytest.fixture()
def apply_migrations(pg_urls):
    """Run alembic upgrade head against the test container."""
    from alembic import command
    from alembic.config import Config

    _, sync_url = pg_urls
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", sync_url)
    command.upgrade(cfg, "head")
    yield
    # Leave schema in place for inspections; container torn down by session fixture


@pytest.fixture()
def reset_db_engine(pg_urls, monkeypatch):
    """Point SQLAlchemy engine singletons at the test container."""
    async_url, sync_url = pg_urls
    monkeypatch.setenv("POSTGRES_URL", async_url)
    monkeypatch.setenv("POSTGRES_URL_SYNC", sync_url)

    import audit_agent.config as cfg_mod
    import audit_agent.storage.postgres as pg_mod

    cfg_mod.reset_settings()
    # Reset SQLAlchemy singletons so they rebuild with the new URL
    pg_mod._engine = None
    pg_mod._session_factory = None

    yield

    cfg_mod.reset_settings()
    pg_mod._engine = None
    pg_mod._session_factory = None
