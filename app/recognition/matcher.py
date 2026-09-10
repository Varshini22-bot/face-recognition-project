"""Cosine-similarity matching for stored face embeddings."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from app.recognition.threshold import SimilarityThreshold
from app.storage.face_repository import PersonRecord


class MatchingError(ValueError):
	"""Raised when an embedding cannot be safely compared."""


@dataclass(frozen=True)
class EmbeddingCandidate:
	"""Identity metadata and an embedding supplied to the matcher."""

	person_id: int
	name: str
	embedding: np.ndarray

	@classmethod
	def from_person(cls, person: PersonRecord) -> "EmbeddingCandidate":
		"""Create a matching candidate from a repository record."""
		return cls(person.id, person.name, person.embedding)


@dataclass(frozen=True)
class RecognitionResult:
	"""The public matching decision without exposing embedding data."""

	recognized: bool
	person_id: int | None
	name: str | None
	similarity: float | None


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
	"""Return cosine similarity, rejecting invalid or zero-norm vectors."""
	first_vector = _as_vector(first)
	second_vector = _as_vector(second)
	if first_vector.shape != second_vector.shape:
		raise MatchingError("embedding dimensions must match")
	first_norm = np.linalg.norm(first_vector)
	second_norm = np.linalg.norm(second_vector)
	if first_norm == 0.0 or second_norm == 0.0:
		raise MatchingError("zero-norm embeddings cannot be compared")
	return float(np.dot(first_vector, second_vector) / (first_norm * second_norm))


class EmbeddingMatcher:
	"""Find the best registered embedding and apply a recognition threshold."""

	def __init__(self, threshold: SimilarityThreshold | None = None) -> None:
		self._threshold = threshold or SimilarityThreshold()

	def match(
		self,
		query_embedding: np.ndarray,
		candidates: Iterable[EmbeddingCandidate | PersonRecord],
	) -> RecognitionResult:
		"""Return the highest-scoring candidate or an unknown result."""
		query = _as_vector(query_embedding)
		best_candidate: EmbeddingCandidate | None = None
		best_similarity: float | None = None
		for candidate_value in candidates:
			candidate = (
				EmbeddingCandidate.from_person(candidate_value)
				if isinstance(candidate_value, PersonRecord)
				else candidate_value
			)
			similarity = cosine_similarity(query, candidate.embedding)
			if best_similarity is None or similarity > best_similarity:
				best_candidate = candidate
				best_similarity = similarity

		if best_candidate is None or best_similarity is None:
			return RecognitionResult(False, None, None, None)
		recognized = self._threshold.is_recognized(best_similarity)
		return RecognitionResult(
			recognized,
			best_candidate.person_id if recognized else None,
			best_candidate.name if recognized else None,
			best_similarity,
		)


def _as_vector(embedding: np.ndarray) -> np.ndarray:
	vector = np.asarray(embedding, dtype=np.float64)
	if vector.ndim != 1 or vector.size == 0:
		raise MatchingError("embedding must be a non-empty one-dimensional vector")
	if not np.isfinite(vector).all():
		raise MatchingError("embedding must contain only finite values")
	return vector
