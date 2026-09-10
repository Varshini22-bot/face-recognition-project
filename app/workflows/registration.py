"""Workflow for registering one person's face and embedding."""

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np

from app.config import AppConfig
from app.detection.face_detector import FaceDetector
from app.embeddings.embedding_generator import EmbeddingGenerator
from app.storage.face_repository import (
	DuplicatePersonError,
	FaceRepository,
	PersonRecord,
)


class RegistrationError(RuntimeError):
	"""Base error for registration failures."""


class InvalidRegistrationImageError(RegistrationError):
	"""Raised when the supplied image cannot be used for registration."""


class NoFaceDetectedError(RegistrationError):
	"""Raised when registration finds no face."""


class MultipleFacesDetectedError(RegistrationError):
	"""Raised when registration finds more than one face."""


@dataclass(frozen=True)
class RegistrationResult:
	"""Result returned after a successful registration."""

	person: PersonRecord


class RegistrationWorkflow:
	"""Coordinate detection, ArcFace embedding, image storage, and persistence."""

	def __init__(
		self,
		detector: FaceDetector,
		embedding_generator: EmbeddingGenerator,
		repository: FaceRepository,
		config: AppConfig | None = None,
	) -> None:
		self._detector = detector
		self._embedding_generator = embedding_generator
		self._repository = repository
		self._config = config or AppConfig.from_environment()

	def register(self, name: str, image_path: str | Path) -> RegistrationResult:
		"""Register exactly one face from ``image_path`` under ``name``."""
		clean_name = name.strip()
		if not clean_name:
			raise RegistrationError("Person name must not be empty")
		input_path = Path(image_path)
		if not input_path.is_file():
			raise InvalidRegistrationImageError(f"Image does not exist: {input_path}")

		try:
			image, faces = self._detector.detect_image(input_path)
		except (OSError, ValueError) as error:
			raise InvalidRegistrationImageError(str(error)) from error
		if len(faces) == 0:
			raise NoFaceDetectedError("No face detected; registration requires exactly one face")
		if len(faces) != 1:
			raise MultipleFacesDetectedError(
				f"Detected {len(faces)} faces; registration requires exactly one face"
			)

		face_x, face_y, face_width, face_height = faces[0]
		face_crop = image[face_y : face_y + face_height, face_x : face_x + face_width]
		if face_crop.size == 0:
			raise InvalidRegistrationImageError("Detected face has an invalid image region")
		try:
			embedding = self._embedding_generator.generate_embedding(
				face_crop, detected_face=True
			)
		except Exception as error:
			raise RegistrationError("Could not generate an embedding for the face") from error

		self._config.registered_faces_dir.mkdir(parents=True, exist_ok=True)
		stored_path = self._unique_image_path(clean_name, input_path.suffix)
		shutil.copy2(input_path, stored_path)
		try:
			person = self._repository.create_person(
				clean_name,
				self._relative_image_path(stored_path),
				embedding,
			)
		except DuplicatePersonError:
			stored_path.unlink(missing_ok=True)
			raise RegistrationError(
				f"A person named '{clean_name}' is already registered"
			) from None
		except Exception:
			stored_path.unlink(missing_ok=True)
			raise
		return RegistrationResult(person)

	def _unique_image_path(self, name: str, suffix: str) -> Path:
		safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "person"
		safe_suffix = suffix.lower() if suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"} else ".png"
		return self._config.registered_faces_dir / f"{safe_name}_{uuid4().hex}{safe_suffix}"

	@staticmethod
	def _relative_image_path(path: Path) -> Path:
		try:
			return path.relative_to(Path(__file__).resolve().parents[2])
		except ValueError:
			return path
