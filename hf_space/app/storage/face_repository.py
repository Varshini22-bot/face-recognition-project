"""Persistence for registered people and their face embeddings."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.storage.database import Database


@dataclass(frozen=True)
class PersonRecord:
	"""A registered person's metadata and decoded embedding."""

	id: int
	name: str
	image_path: Path
	embedding: np.ndarray
	created_at: str


class DuplicatePersonError(ValueError):
	"""Raised when a person name is already registered."""


class PersonNotFoundError(LookupError):
	"""Raised when a requested person does not exist."""


class FaceRepository:
	"""Store and retrieve registered people using parameterized SQLite queries."""

	def __init__(self, database: Database) -> None:
		self._database = database
		self._database.initialize()

	def create_person(
		self,
		name: str,
		image_path: str | Path,
		embedding: np.ndarray,
		created_at: str | None = None,
	) -> PersonRecord:
		"""Persist one person and serialize its embedding as NumPy bytes."""
		clean_name = name.strip()
		if not clean_name:
			raise ValueError("name must not be empty")
		vector = self._validate_embedding(embedding)
		timestamp = created_at or datetime.now(timezone.utc).isoformat()
		try:
			with self._database.connection() as connection:
				cursor = connection.execute(
					"""
					INSERT INTO persons
					(name, image_path, embedding, embedding_dtype, embedding_dimension, created_at)
					VALUES (?, ?, ?, ?, ?, ?)
					""",
					(
						clean_name,
						str(image_path),
						vector.tobytes(),
						str(vector.dtype),
						vector.size,
						timestamp,
					),
				)
				person_id = int(cursor.lastrowid)
		except sqlite3.IntegrityError as error:
			if "UNIQUE" in str(error).upper():
				raise DuplicatePersonError(f"A person named '{clean_name}' is already registered") from error
			raise
		return PersonRecord(person_id, clean_name, Path(image_path), vector.copy(), timestamp)

	def get_person_by_id(self, person_id: int) -> PersonRecord:
		"""Return a person by ID or raise ``PersonNotFoundError``."""
		with self._database.connection() as connection:
			row = connection.execute(
				"SELECT * FROM persons WHERE id = ?", (person_id,)
			).fetchone()
		return self._row_to_record(row)

	def get_person_by_name(self, name: str) -> PersonRecord:
		"""Return a person by case-insensitive name."""
		with self._database.connection() as connection:
			row = connection.execute(
				"SELECT * FROM persons WHERE name = ? COLLATE NOCASE", (name.strip(),)
			).fetchone()
		return self._row_to_record(row)

	def get_all_people(self) -> list[PersonRecord]:
		"""Return all registered people ordered by ID."""
		with self._database.connection() as connection:
			rows = connection.execute("SELECT * FROM persons ORDER BY id").fetchall()
		return [self._row_to_record(row) for row in rows]

	def delete_person(self, person_id: int) -> None:
		"""Delete a person by ID or raise ``PersonNotFoundError``."""
		with self._database.connection() as connection:
			cursor = connection.execute("DELETE FROM persons WHERE id = ?", (person_id,))
			if cursor.rowcount == 0:
				raise PersonNotFoundError(f"No person exists with ID {person_id}")

	@staticmethod
	def _validate_embedding(embedding: np.ndarray) -> np.ndarray:
		vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
		if vector.size == 0 or not np.isfinite(vector).all():
			raise ValueError("embedding must be a non-empty finite NumPy vector")
		return vector

	@classmethod
	def _row_to_record(cls, row: sqlite3.Row | tuple | None) -> PersonRecord:
		if row is None:
			raise PersonNotFoundError("Registered person was not found")
		person_id, name, image_path, data, dtype, dimension, created_at = row
		vector = np.frombuffer(data, dtype=np.dtype(dtype)).copy()
		if vector.size != dimension:
			raise ValueError("Stored embedding dimension does not match its data")
		return PersonRecord(int(person_id), name, Path(image_path), vector, created_at)
