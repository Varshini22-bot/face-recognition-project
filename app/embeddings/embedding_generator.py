"""ArcFace embedding generation through the supported DeepFace API."""

from pathlib import Path
from typing import Any, Callable, TypeAlias

import numpy as np


ImageInput: TypeAlias = str | Path | np.ndarray
RepresentFunction: TypeAlias = Callable[..., list[dict[str, Any]]]


class EmbeddingGenerationError(RuntimeError):
	"""Base error for failures while generating a face embedding."""


class FaceNotDetectedError(EmbeddingGenerationError):
	"""Raised when DeepFace cannot find a face in the input."""


class MultipleFacesDetectedError(EmbeddingGenerationError):
	"""Raised when an input contains more than one face."""


class EmbeddingGenerator:
	"""Generate one ArcFace embedding from a face image or detected crop."""

	MODEL_NAME = "ArcFace"

	def __init__(
		self,
		detector_backend: str = "yunet",
		represent_function: RepresentFunction | None = None,
	) -> None:
		"""Configure ArcFace representation and the DeepFace detector backend."""
		if not detector_backend:
			raise ValueError("detector_backend must not be empty")
		self._detector_backend = detector_backend
		self._represent_function = represent_function

	def generate_embedding(
		self,
		image: ImageInput,
		detected_face: bool = False,
	) -> np.ndarray:
		"""Return one finite ``float32`` ArcFace embedding.

		Set ``detected_face=True`` when ``image`` is already a single face crop
		from the Phase 2 detector; otherwise DeepFace detects the face first.
		"""
		self._validate_image_input(image)
		detector_backend = "skip" if detected_face else self._detector_backend
		try:
			results = self._represent(
				image,
				model_name=self.MODEL_NAME,
				enforce_detection=True,
				detector_backend=detector_backend,
				align=True,
				normalization="ArcFace",
				l2_normalize=True,
			)
		except Exception as error:
			if "face" in str(error).lower() and "detect" in str(error).lower():
				raise FaceNotDetectedError("No face was detected in the input image") from error
			raise EmbeddingGenerationError("DeepFace could not generate an ArcFace embedding") from error

		if not results:
			raise FaceNotDetectedError("No face was detected in the input image")
		if len(results) != 1:
			raise MultipleFacesDetectedError(
				f"Expected one face, but detected {len(results)} faces"
			)
		try:
			embedding = np.asarray(results[0]["embedding"], dtype=np.float32).reshape(-1)
		except (KeyError, TypeError, ValueError) as error:
			raise EmbeddingGenerationError("DeepFace returned an invalid embedding") from error
		if embedding.size == 0 or not np.isfinite(embedding).all():
			raise EmbeddingGenerationError("DeepFace returned an invalid embedding")
		return embedding

	def _represent(self, image: ImageInput, **kwargs: Any) -> list[dict[str, Any]]:
		if self._represent_function is not None:
			return self._represent_function(image, **kwargs)
		from deepface import DeepFace

		return DeepFace.represent(image, **kwargs)

	@staticmethod
	def _validate_image_input(image: ImageInput) -> None:
		if isinstance(image, (str, Path)):
			if not Path(image).is_file():
				raise FileNotFoundError(f"Could not read image: {image}")
			return
		if not isinstance(image, np.ndarray) or image.size == 0:
			raise ValueError("image must be a non-empty image path or NumPy array")
		if image.ndim not in (2, 3):
			raise ValueError("image must be grayscale or a color image")
