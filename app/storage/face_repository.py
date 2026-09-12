"""Persistence for registered people and their face embeddings."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import asyncpg

from app.storage.database import Database


@dataclass(frozen=True)
class PersonRecord:
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
    """Store and retrieve registered people using parameterized Neon queries."""

    def __init__(self, database: Database) -> None:
        self._database = database
        self._database.initialize()

    def create_person(self, name: str, image_path: str | Path, embedding: np.ndarray, created_at: str | None = None) -> PersonRecord:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("name must not be empty")
        vector = self._validate_embedding(embedding)
        timestamp = created_at or datetime.now(timezone.utc).isoformat()
        try:
            row = self._database.run(self._insert(clean_name, image_path, vector, timestamp))
        except asyncpg.UniqueViolationError as error:
            raise DuplicatePersonError(f"A person named '{clean_name}' is already registered") from error
        return self._row_to_record(row)

    async def _insert(self, name: str, image_path: str | Path, vector: np.ndarray, timestamp: str):
        async with self._database.connection() as connection:
            return await connection.fetchrow(
                """INSERT INTO persons (name, image_path, embedding, embedding_dtype, embedding_dimension, created_at)
                   VALUES ($1, $2, $3, $4, $5, $6) RETURNING id, name, image_path, embedding, embedding_dtype, embedding_dimension, created_at""",
                name, str(image_path), vector.tobytes(), str(vector.dtype), vector.size, timestamp,
            )

    def get_person_by_id(self, person_id: int) -> PersonRecord:
        return self._get_one("SELECT id, name, image_path, embedding, embedding_dtype, embedding_dimension, created_at FROM persons WHERE id = $1", person_id)

    def get_person_by_name(self, name: str) -> PersonRecord:
        return self._get_one("SELECT id, name, image_path, embedding, embedding_dtype, embedding_dimension, created_at FROM persons WHERE lower(name) = lower($1)", name.strip())

    def get_all_people(self) -> list[PersonRecord]:
        rows = self._database.run(self._all())
        return [self._row_to_record(row) for row in rows]

    async def _all(self):
        async with self._database.connection() as connection:
            return await connection.fetch("SELECT id, name, image_path, embedding, embedding_dtype, embedding_dimension, created_at FROM persons ORDER BY id")

    def _get_one(self, query: str, value: object) -> PersonRecord:
        row = self._database.run(self._fetch_one(query, value))
        return self._row_to_record(row)

    async def _fetch_one(self, query: str, value: object):
        async with self._database.connection() as connection:
            return await connection.fetchrow(query, value)

    def delete_person(self, person_id: int) -> None:
        deleted = self._database.run(self._delete(person_id))
        if deleted == 0:
            raise PersonNotFoundError(f"No person exists with ID {person_id}")

    async def _delete(self, person_id: int) -> int:
        async with self._database.connection() as connection:
            result = await connection.execute("DELETE FROM persons WHERE id = $1", person_id)
            return int(result.rsplit(" ", 1)[-1])

    @staticmethod
    def _validate_embedding(embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("embedding must be a non-empty finite NumPy vector")
        return vector

    @classmethod
    def _row_to_record(cls, row) -> PersonRecord:
        if row is None:
            raise PersonNotFoundError("Registered person was not found")
        vector = np.frombuffer(bytes(row["embedding"]), dtype=np.dtype(row["embedding_dtype"])).copy()
        if vector.size != row["embedding_dimension"]:
            raise ValueError("Stored embedding dimension does not match its data")
        return PersonRecord(int(row["id"]), row["name"], Path(row["image_path"]), vector, row["created_at"])
