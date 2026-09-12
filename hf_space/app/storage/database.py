"""SQLite connection and schema management."""

import sqlite3
from pathlib import Path
from typing import Iterator
from contextlib import contextmanager


SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    image_path TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedding_dtype TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    created_at TEXT NOT NULL
)
"""


class Database:
	"""Manage short-lived SQLite connections and schema initialization."""

	def __init__(self, database_path: str | Path) -> None:
		self.database_path = Path(database_path)

	def initialize(self) -> None:
		"""Create the database directory and persons table when needed."""
		self.database_path.parent.mkdir(parents=True, exist_ok=True)
		with self.connection() as connection:
			connection.execute(SCHEMA)

	@contextmanager
	def connection(self) -> Iterator[sqlite3.Connection]:
		"""Yield a connection that commits on success and rolls back on failure."""
		self.database_path.parent.mkdir(parents=True, exist_ok=True)
		connection = sqlite3.connect(self.database_path)
		try:
			yield connection
			connection.commit()
		except Exception:
			connection.rollback()
			raise
		finally:
			connection.close()
