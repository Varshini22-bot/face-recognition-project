"""End-to-end recognition of faces in a still image."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.detection.face_detector import FaceBox, FaceDetector
from app.embeddings.embedding_generator import EmbeddingGenerator
from app.recognition.matcher import EmbeddingMatcher
from app.storage.face_repository import FaceRepository


class ImageRecognitionError(RuntimeError):
	"""Raised when an image cannot be processed for recognition."""


@dataclass(frozen=True)
class FaceRecognitionResult:
	"""Recognition information for one detected face."""

	box: FaceBox
	recognized: bool
	person_id: int | None
	name: str | None
	similarity: float | None


@dataclass(frozen=True)
class ImageRecognitionReport:
	"""Recognition results for every face in one image."""

	image_path: Path
	faces: tuple[FaceRecognitionResult, ...]
	annotated_image: np.ndarray | None = None


class ImageRecognitionWorkflow:
	"""Connect face detection, ArcFace embeddings, storage, and matching."""

	def __init__(
		self,
		detector: FaceDetector,
		embedding_generator: EmbeddingGenerator,
		repository: FaceRepository,
		matcher: EmbeddingMatcher,
	) -> None:
		self._detector = detector
		self._embedding_generator = embedding_generator
		self._repository = repository
		self._matcher = matcher

	def recognize(
		self,
		image_path: str | Path,
		output_path: str | Path | None = None,
		show_similarity: bool = True,
	) -> ImageRecognitionReport:
		"""Recognize every face and optionally save an annotated image."""
		input_path = Path(image_path)
		if not input_path.is_file():
			raise ImageRecognitionError(f"Image does not exist: {input_path}")
		try:
			image, boxes = self._detector.detect_image(input_path)
		except (OSError, ValueError) as error:
			raise ImageRecognitionError(str(error)) from error

		people = self._repository.get_all_people()
		face_results: list[FaceRecognitionResult] = []
		for box in boxes:
			crop = self._crop_face(image, box)
			try:
				embedding = self._embedding_generator.generate_embedding(
					crop, detected_face=True
				)
				match = self._matcher.match(embedding, people)
			except Exception as error:
				raise ImageRecognitionError(
					f"Could not recognize face at ({box[0]}, {box[1]})"
				) from error
			face_results.append(
				FaceRecognitionResult(
					box=box,
					recognized=match.recognized,
					person_id=match.person_id,
					name=match.name,
					similarity=match.similarity,
				)
			)

		annotated = None
		if output_path is not None:
			annotated = self.annotate_image(image, face_results, show_similarity)
			destination = Path(output_path)
			destination.parent.mkdir(parents=True, exist_ok=True)
			if not cv2.imwrite(str(destination), annotated):
				raise ImageRecognitionError(f"Could not save annotated image: {destination}")
		return ImageRecognitionReport(input_path, tuple(face_results), annotated)

	@staticmethod
	def annotate_image(
		image: np.ndarray,
		faces: list[FaceRecognitionResult] | tuple[FaceRecognitionResult, ...],
		show_similarity: bool = True,
	) -> np.ndarray:
		"""Return a copy with green known-face or red unknown-face labels."""
		annotated = image.copy()
		for face in faces:
			x, y, width, height = face.box
			color = (0, 180, 0) if face.recognized else (0, 0, 255)
			label = face.name if face.recognized and face.name else "Unknown"
			if show_similarity and face.similarity is not None:
				label = f"{label} ({face.similarity:.2f})"
			cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 2)
			cv2.putText(
				annotated,
				label,
				(x, max(y - 10, 20)),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.6,
				color,
				2,
				cv2.LINE_AA,
			)
		return annotated

	@staticmethod
	def _crop_face(image: np.ndarray, box: FaceBox) -> np.ndarray:
		x, y, width, height = box
		image_height, image_width = image.shape[:2]
		left = max(0, x)
		top = max(0, y)
		right = min(image_width, x + width)
		bottom = min(image_height, y + height)
		crop = image[top:bottom, left:right]
		if crop.size == 0:
			raise ImageRecognitionError(f"Detected face has an invalid box: {box}")
		return crop
