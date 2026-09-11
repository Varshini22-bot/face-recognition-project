import numpy as np
import pytest

from app.recognition.matcher import (
	EmbeddingCandidate,
	EmbeddingMatcher,
	MatchingError,
	cosine_similarity,
)
from app.recognition.threshold import SimilarityThreshold


def candidate(person_id: int, name: str, values: list[float]) -> EmbeddingCandidate:
	return EmbeddingCandidate(person_id, name, np.asarray(values, dtype=np.float32))


def test_identical_embeddings_have_similarity_one() -> None:
	assert cosine_similarity(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == pytest.approx(1.0)


def test_orthogonal_embeddings_have_similarity_zero() -> None:
	assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_opposite_embeddings_have_similarity_minus_one() -> None:
	assert cosine_similarity(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == pytest.approx(-1.0)


def test_best_match_is_selected() -> None:
	result = EmbeddingMatcher(SimilarityThreshold(0.8)).match(
		np.array([1.0, 0.0]),
		[candidate(1, "Other", [0.0, 1.0]), candidate(2, "Ada", [1.0, 0.0])],
	)

	assert result.recognized is True
	assert result.person_id == 2
	assert result.name == "Ada"
	assert result.similarity == pytest.approx(1.0)


def test_similarity_below_threshold_is_unknown() -> None:
	result = EmbeddingMatcher(SimilarityThreshold(0.8)).match(
		np.array([1.0, 0.0]), [candidate(1, "Ada", [0.7, 0.71414284])]
	)

	assert result.recognized is False
	assert result.person_id is None
	assert result.name is None
	assert result.similarity is not None
	assert result.similarity < 0.8


def test_similarity_at_threshold_is_recognized() -> None:
	result = EmbeddingMatcher(SimilarityThreshold(0.50)).match(
		np.array([1.0, 0.0]),
		[EmbeddingCandidate(1, "Ada", np.array([0.50, 0.8660254], dtype=np.float64))],
	)

	assert result.recognized is True


def test_dissimilar_embeddings_are_not_recognized() -> None:
	result = EmbeddingMatcher(SimilarityThreshold(0.50)).match(
		np.array([1.0, 0.0]),
		[candidate(1, "Ada", [0.0, 1.0])],
	)

	assert result.recognized is False
	assert result.name is None
	assert result.similarity == pytest.approx(0.0)


def test_invalid_empty_embedding_is_rejected() -> None:
	with pytest.raises(MatchingError, match="non-empty"):
		EmbeddingMatcher().match(np.array([], dtype=np.float32), [])


def test_invalid_non_finite_embedding_is_rejected() -> None:
	with pytest.raises(MatchingError, match="finite"):
		cosine_similarity(np.array([np.nan]), np.array([1.0]))


def test_default_threshold_is_0_50() -> None:
	assert SimilarityThreshold().value == pytest.approx(0.50)


def test_similarity_above_threshold_is_recognized() -> None:
	result = EmbeddingMatcher(SimilarityThreshold(0.8)).match(
		np.array([1.0, 0.0]), [candidate(1, "Ada", [1.0, 0.1])]
	)

	assert result.recognized is True


def test_empty_database_returns_unknown_without_score() -> None:
	result = EmbeddingMatcher().match(np.array([1.0, 0.0]), [])

	assert result == result.__class__(False, None, None, None)


def test_invalid_dimensions_are_rejected() -> None:
	with pytest.raises(MatchingError, match="dimensions"):
		EmbeddingMatcher().match(np.array([1.0, 0.0]), [candidate(1, "Ada", [1.0])])


def test_zero_norm_vectors_are_rejected() -> None:
	with pytest.raises(MatchingError, match="zero-norm"):
		cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 0.0]))


def test_result_does_not_expose_embedding_data() -> None:
	result = EmbeddingMatcher().match(
		np.array([1.0, 0.0]), [candidate(1, "Ada", [1.0, 0.0])]
	)

	assert not hasattr(result, "embedding")
	assert "embedding" not in result.__dict__
