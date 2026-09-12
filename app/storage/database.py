"""Neon PostgreSQL connection and schema management."""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import asyncpg



SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    image_path TEXT NOT NULL,
    embedding BYTEA NOT NULL,
    embedding_dtype TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    created_at TEXT NOT NULL
)
"""


class Database:
    """Manage Neon PostgreSQL connections for serverless requests."""

    def __init__(self, database_path: str | os.PathLike[str] | None = None) -> None:
        self.database_url = os.environ.get("DATABASE_URL")
        if not self.database_url:
            raise RuntimeError("DATABASE_URL is required for Neon persistence")

    def initialize(self) -> None:
        asyncio.run(self._initialize())

    async def _initialize(self) -> None:
        async with self.connection() as connection:
            await connection.execute(SCHEMA)

    @staticmethod
    def _connection_options(database_url: str) -> tuple[str, dict[str, object]]:
        parsed = urlsplit(database_url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        sslmode = query.pop("sslmode", None)
        query.pop("channel_binding", None)
        normalized_url = urlunsplit(parsed._replace(query=urlencode(query)))
        options: dict[str, object] = {}
        if sslmode and sslmode != "disable":
            options["ssl"] = "require"
        return normalized_url, options

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[asyncpg.Connection]:
        normalized_url, options = self._connection_options(self.database_url)
        connection = await asyncpg.connect(normalized_url, **options)
        try:
            yield connection
        finally:
            await connection.close()

    def run(self, operation: Any) -> Any:
        """Run one async database operation from the existing sync repository API."""
        return asyncio.run(operation)
