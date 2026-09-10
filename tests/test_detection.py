import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from app.detection.face_detector import FaceDetector


def test_detect_faces_returns_boxes_from_yunet(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setattr(FaceDetector, "_ensure_model", lambda model_path: None)
	monkeypatch.setattr("app.detection.face_detector.cv2.FaceDetectorYN_create", lambda *args: object())
	detector = FaceDetector(model_path="fake.onnx")
	expected_boxes = [(10, 20, 30, 40)]

	class FakeYuNet:
		def setInputSize(self, size):
			pass

		def setScoreThreshold(self, threshold):
			pass

		def detect(self, image):
			return None, np.array([[*expected_boxes[0], 0, 0, 0, 0, 0, 0, 0, 0, 0]])

	monkeypatch.setattr(detector, "_detector", FakeYuNet())

	image = np.zeros((100, 100, 3), dtype=np.uint8)
	assert detector.detect_faces(image) == expected_boxes


def test_draw_boxes_does_not_modify_original_image() -> None:
	image = np.zeros((80, 80, 3), dtype=np.uint8)
	detector = FaceDetector()

	annotated = detector.draw_boxes(image, [(10, 10, 20, 20)])

	assert np.array_equal(image, np.zeros((80, 80, 3), dtype=np.uint8))
	assert not np.array_equal(annotated, image)


def test_detect_faces_rejects_empty_images() -> None:
	detector = object.__new__(FaceDetector)

	with pytest.raises(ValueError, match="non-empty"):
		detector.detect_faces(np.array([]))
