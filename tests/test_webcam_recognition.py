import cv2
import numpy as np
import pytest

from app.recognition.matcher import RecognitionResult
from app.workflows.webcam_recognition import (
	WebcamRecognitionError,
	WebcamRecognitionWorkflow,
)


class FakeCapture:
	def __init__(self, opened=True, frames=None):
		self.opened = opened
		self.frames = list(frames or [])
		self.released = False

	def isOpened(self):
		return self.opened

	def read(self):
		if not self.frames:
			return False, None
		return self.frames.pop(0)

	def release(self):
		self.released = True


class FakeDetector:
	def __init__(self, boxes):
		self.boxes = boxes

	def detect_faces(self, frame):
		return self.boxes


class FakeEmbeddingGenerator:
	def __init__(self):
		self.calls = 0

	def generate_embedding(self, crop, detected_face=False):
		self.calls += 1
		assert detected_face is True
		return np.array([float(self.calls), 0.0], dtype=np.float32)


class FakeRepository:
	def __init__(self, people=None):
		self.people = people or []
		self.calls = 0

	def get_all_people(self):
		self.calls += 1
		return self.people


class FakeMatcher:
	def __init__(self, results):
		self.results = iter(results)
		self.calls = 0

	def match(self, embedding, people):
		self.calls += 1
		return next(self.results)


def make_workflow(boxes, results, people=None):
	return WebcamRecognitionWorkflow(
		FakeDetector(boxes),
		FakeEmbeddingGenerator(),
		FakeRepository(people),
		FakeMatcher(results),
	)


def patch_display(monkeypatch, wait_keys):
	monkeypatch.setattr(cv2, "imshow", lambda *args: None)
	monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
	monkeypatch.setattr(cv2, "waitKey", lambda delay: wait_keys.pop(0))


def test_camera_open_failure_releases_and_raises(monkeypatch):
	capture = FakeCapture(opened=False)
	workflow = make_workflow([], [])

	with pytest.raises(WebcamRecognitionError, match="Could not open webcam"):
		workflow.run(capture_factory=lambda index: capture)

	assert capture.released is True


def test_camera_read_failure_exits_cleanly(monkeypatch):
	capture = FakeCapture(frames=[(False, None)])
	workflow = make_workflow([], [])
	patch_display(monkeypatch, [])

	report = workflow.run(capture_factory=lambda index: capture)

	assert report.frames_processed == 0
	assert capture.released is True


def test_recognized_face_is_processed_and_displayed(monkeypatch):
	frame = np.zeros((80, 80, 3), dtype=np.uint8)
	capture = FakeCapture(frames=[(True, frame)])
	workflow = make_workflow(
		[(10, 10, 20, 20)], [RecognitionResult(True, 4, "Alice", 0.95)], people=[object()]
	)
	patch_display(monkeypatch, [ord("q")])

	report = workflow.run(capture_factory=lambda index: capture)

	assert report.frames_processed == 1
	assert report.stopped_by_user is True


def test_unknown_face_has_no_identity(monkeypatch):
	frame = np.zeros((80, 80, 3), dtype=np.uint8)
	workflow = make_workflow(
		[(10, 10, 20, 20)], [RecognitionResult(False, None, None, 0.2)], people=[object()]
	)
	results, _ = workflow.process_frame(frame, [object()])

	assert results[0].recognized is False
	assert results[0].name is None
	assert results[0].person_id is None


def test_multiple_faces_are_processed_independently():
	frame = np.zeros((100, 100, 3), dtype=np.uint8)
	boxes = [(1, 2, 10, 10), (30, 40, 20, 20)]
	workflow = make_workflow(
		boxes,
		[RecognitionResult(True, 1, "Alice", 0.9), RecognitionResult(False, None, None, 0.3)],
		people=[object()],
	)

	results, _ = workflow.process_frame(frame, [object()])

	assert [result.box for result in results] == boxes
	assert [result.name for result in results] == ["Alice", None]


def test_registered_people_load_once_for_multiple_frames(monkeypatch):
	frame = np.zeros((80, 80, 3), dtype=np.uint8)
	capture = FakeCapture(frames=[(True, frame), (True, frame)])
	workflow = make_workflow(
		[(10, 10, 20, 20)],
		[RecognitionResult(False, None, None, None), RecognitionResult(False, None, None, None)],
	)
	patch_display(monkeypatch, [-1, ord("q")])

	workflow.run(capture_factory=lambda index: capture)

	assert workflow._repository.calls == 1


def test_q_exits_and_resources_are_cleaned(monkeypatch):
	frame = np.zeros((80, 80, 3), dtype=np.uint8)
	capture = FakeCapture(frames=[(True, frame)])
	workflow = make_workflow([], [])
	patch_display(monkeypatch, [ord("q")])

	report = workflow.run(capture_factory=lambda index: capture)

	assert report.stopped_by_user is True
	assert capture.released is True


def test_results_do_not_expose_embeddings():
	workflow = make_workflow([(1, 1, 10, 10)], [RecognitionResult(True, 1, "Alice", 0.9)])
	results, _ = workflow.process_frame(np.zeros((30, 30, 3), dtype=np.uint8), [object()])

	assert "embedding" not in results[0].__dict__