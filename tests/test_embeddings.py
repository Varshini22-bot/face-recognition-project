import numpy as np
import pytest

from app.embeddings.embedding_generator import (
	EmbeddingGenerator,
	EmbeddingGenerationError,
	FaceNotDetectedError,
	MultipleFacesDetectedError,
)


def test_generate_embedding_uses_arcface_and_returns_float32() -> None:
	represent_calls = []

	def fake_represent(image, **kwargs):
		represent_calls.append((image, kwargs))
		return [{"embedding": [0.1, 0.2, 0.3]}]

	generator = EmbeddingGenerator(represent_function=fake_represent)
	embedding = generator.generate_embedding(np.zeros((20, 20, 3), dtype=np.uint8))

	assert embedding.dtype == np.float32
	assert embedding.shape == (3,)
	assert np.allclose(embedding, [0.1, 0.2, 0.3])
	assert represent_calls[0][1]["model_name"] == "ArcFace"
	assert represent_calls[0][1]["detector_backend"] == "yunet"


def test_detected_crop_uses_skip_backend() -> None:
	used_backends = []

	def fake_represent(image, **kwargs):
		used_backends.append(kwargs["detector_backend"])
		return [{"embedding": [1.0, 2.0]}]

	generator = EmbeddingGenerator(represent_function=fake_represent)
	generator.generate_embedding(np.zeros((20, 20, 3), dtype=np.uint8), detected_face=True)

	assert used_backends == ["skip"]


def test_no_face_raises_specific_error() -> None:
	generator = EmbeddingGenerator(represent_function=lambda image, **kwargs: [])

	with pytest.raises(FaceNotDetectedError):
		generator.generate_embedding(np.zeros((20, 20, 3), dtype=np.uint8))


def test_multiple_faces_are_rejected() -> None:
	generator = EmbeddingGenerator(
		represent_function=lambda image, **kwargs: [
			{"embedding": [1.0]},
			{"embedding": [2.0]},
		]
	)

	with pytest.raises(MultipleFacesDetectedError):
		generator.generate_embedding(np.zeros((20, 20, 3), dtype=np.uint8))


def test_invalid_embedding_is_rejected() -> None:
	generator = EmbeddingGenerator(
		represent_function=lambda image, **kwargs: [{"embedding": [np.nan]}]
	)

	with pytest.raises(EmbeddingGenerationError):
		generator.generate_embedding(np.zeros((20, 20, 3), dtype=np.uint8))
