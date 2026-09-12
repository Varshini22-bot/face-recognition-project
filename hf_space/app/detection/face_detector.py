"""Face detection using OpenCV YuNet, the detector used by DeepFace."""

from pathlib import Path
from typing import TypeAlias
from urllib.request import urlopen

import cv2
import numpy as np


FaceBox: TypeAlias = tuple[int, int, int, int]
YUNET_MODEL_NAME = "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_URL = (
	"https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
	f"{YUNET_MODEL_NAME}"
)


class FaceDetector:
	"""Detect and annotate faces without performing face recognition."""

	def __init__(
		self,
		model_path: str | Path | None = None,
		score_threshold: float = 0.9,
	) -> None:
		"""Create a YuNet detector compatible with OpenCV 5 and DeepFace."""
		if not 0.0 < score_threshold <= 1.0:
			raise ValueError("score_threshold must be between 0 and 1")
		if not hasattr(cv2, "FaceDetectorYN_create"):
			raise RuntimeError("This OpenCV build does not provide the YuNet detector API")

		model_file = Path(model_path) if model_path else self._cached_model_path()
		FaceDetector._ensure_model(model_file)
		try:
			self._detector = cv2.FaceDetectorYN_create(str(model_file), "", (0, 0))
		except Exception as error:
			raise RuntimeError(f"Could not load YuNet model: {model_file}") from error
		self._score_threshold = score_threshold

	@staticmethod
	def _cached_model_path() -> Path:
		return Path.home() / ".deepface" / "weights" / YUNET_MODEL_NAME

	@staticmethod
	def _ensure_model(model_path: Path) -> None:
		if model_path.is_file():
			return
		model_path.parent.mkdir(parents=True, exist_ok=True)
		try:
			with urlopen(YUNET_MODEL_URL, timeout=30) as response:
				model_path.write_bytes(response.read())
		except Exception as error:
			model_path.unlink(missing_ok=True)
			raise OSError(
				f"Could not download YuNet model to {model_path}. "
				f"Download it from {YUNET_MODEL_URL}."
			) from error

	def detect_faces(self, image: np.ndarray) -> list[FaceBox]:
		"""Return face boxes as ``(x, y, width, height)`` tuples."""
		if image is None or image.size == 0:
			raise ValueError("image must be a non-empty NumPy array")
		if image.ndim not in (2, 3):
			raise ValueError("image must be grayscale or a color image")

		if image.ndim == 2:
			image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
		height, width = image.shape[:2]
		self._detector.setInputSize((width, height))
		self._detector.setScoreThreshold(self._score_threshold)
		_, detected = self._detector.detect(image)
		if detected is None:
			return []
		return [tuple(map(int, face[:4])) for face in detected]

	@staticmethod
	def draw_boxes(
		image: np.ndarray,
		faces: list[FaceBox],
		color: tuple[int, int, int] = (0, 255, 0),
		thickness: int = 2,
	) -> np.ndarray:
		"""Return a copy of ``image`` with a rectangle around every face."""
		annotated = image.copy()
		for x, y, width, height in faces:
			cv2.rectangle(annotated, (x, y), (x + width, y + height), color, thickness)
		return annotated

	def detect_image(self, image_path: str | Path) -> tuple[np.ndarray, list[FaceBox]]:
		"""Load an image, detect its faces, and return both image and boxes."""
		path = Path(image_path)
		image = cv2.imread(str(path))
		if image is None:
			raise FileNotFoundError(f"Could not read image: {path}")
		return image, self.detect_faces(image)
