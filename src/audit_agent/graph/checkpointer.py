"""Postgres checkpointer for LangGraph."""

from __future__ import annotations


async def create_checkpointer(conn_string: str):  # type: ignore[return]
    """Create an AsyncPostgresSaver backed by a psycopg_pool connection pool.

    ``AsyncPostgresSaver.from_conn_string()`` returns a context manager, not the
    saver itself.  Use ``AsyncConnectionPool`` directly so we control the lifetime.
    """
    from psycopg_pool import AsyncConnectionPool
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    pool = AsyncConnectionPool(conn_string, open=False, kwargs={"autocommit": True})
    await pool.open()
    saver = AsyncPostgresSaver(pool)
    await saver.setup()
    return saver
