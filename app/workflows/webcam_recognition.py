"""Real-time webcam face recognition workflow."""

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

import cv2
import numpy as np

from app.detection.face_detector import FaceBox, FaceDetector
from app.embeddings.embedding_generator import EmbeddingGenerator
from app.recognition.matcher import EmbeddingMatcher
from app.storage.face_repository import FaceRepository, PersonRecord
from app.workflows.image_recognition import (
	FaceRecognitionResult,
	ImageRecognitionWorkflow,
)


class WebcamRecognitionError(RuntimeError):
	"""Raised when the webcam cannot be used."""


@dataclass(frozen=True)
class WebcamRunReport:
	"""Non-sensitive summary of a completed webcam session."""

	frames_processed: int
	stopped_by_user: bool


class VideoCaptureProtocol(Protocol):
	def isOpened(self) -> bool: ...

	def read(self) -> tuple[bool, np.ndarray | None]: ...

	def release(self) -> None: ...


CaptureFactory = Callable[[int], VideoCaptureProtocol]


class WebcamRecognitionWorkflow:
	"""Recognize faces continuously while reusing loaded models and people."""

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

	def run(
		self,
		camera_index: int = 0,
		window_name: str = "Face Recognition",
		capture_factory: CaptureFactory = cv2.VideoCapture,
	) -> WebcamRunReport:
		"""Run until ``q`` is pressed or the camera stops returning frames."""
		capture = capture_factory(camera_index)
		if not capture.isOpened():
			capture.release()
			raise WebcamRecognitionError(
				f"Could not open webcam at camera index {camera_index}"
			)

		try:
			people = self._repository.get_all_people()
			frames_processed = 0
			stopped_by_user = False
			while True:
				ok, frame = capture.read()
				if not ok or frame is None:
					break
				results, annotated = self.process_frame(frame, people)
				cv2.imshow(window_name, annotated)
				frames_processed += 1
				if cv2.waitKey(1) & 0xFF == ord("q"):
					stopped_by_user = True
					break
		finally:
			capture.release()
			cv2.destroyAllWindows()
		return WebcamRunReport(frames_processed, stopped_by_user)

	def process_frame(
		self,
		frame: np.ndarray,
		people: Iterable[PersonRecord],
	) -> tuple[tuple[FaceRecognitionResult, ...], np.ndarray]:
		"""Recognize every face in one BGR frame and return annotated output."""
		if frame is None or frame.size == 0:
			raise WebcamRecognitionError("Webcam returned an empty frame")
		try:
			boxes = self._detector.detect_faces(frame)
		except (OSError, ValueError) as error:
			raise WebcamRecognitionError(f"Could not detect faces in webcam frame: {error}") from error

		face_results: list[FaceRecognitionResult] = []
		people_snapshot = tuple(people)
		for box in boxes:
			try:
				crop = self._crop_face(frame, box)
				embedding = self._embedding_generator.generate_embedding(
					crop, detected_face=True
				)
				match = self._matcher.match(embedding, people_snapshot)
				face_results.append(
					FaceRecognitionResult(
						box, match.recognized, match.person_id, match.name, match.similarity
					)
				)
			except Exception:
				face_results.append(FaceRecognitionResult(box, False, None, None, None))

		annotated = ImageRecognitionWorkflow.annotate_image(frame, face_results)
		return tuple(face_results), annotated

	@staticmethod
	def _crop_face(frame: np.ndarray, box: FaceBox) -> np.ndarray:
		x, y, width, height = box
		frame_height, frame_width = frame.shape[:2]
		left = max(0, x)
		top = max(0, y)
		right = min(frame_width, x + width)
		bottom = min(frame_height, y + height)
		crop = frame[top:bottom, left:right]
		if crop.size == 0:
			raise WebcamRecognitionError(f"Detected face has an invalid box: {box}")
		return crop
