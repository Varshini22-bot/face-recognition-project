from pathlib import Path

import cv2
import numpy as np
import pytest

from app.recognition.matcher import RecognitionResult
from app.workflows.image_recognition import (
	FaceRecognitionResult,
	ImageRecognitionError,
	ImageRecognitionWorkflow,
)


class FakeDetector:
	def __init__(self, image: np.ndarray, boxes: list[tuple[int, int, int, int]]):
		self.image = image
		self.boxes = boxes

	def detect_image(self, image_path: Path):
		return self.image, self.boxes


class FakeEmbeddingGenerator:
	def __init__(self):
		self.crops: list[np.ndarray] = []

	def generate_embedding(self, image: np.ndarray, detected_face: bool = False):
		assert detected_face is True
		self.crops.append(image.copy())
		return np.array([float(len(self.crops)), 0.0], dtype=np.float32)


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
		self.embeddings = []

	def match(self, embedding, candidates):
		self.embeddings.append(embedding.copy())
		return next(self.results)


def make_workflow(tmp_path, boxes, results, people=None):
	image = np.zeros((100, 120, 3), dtype=np.uint8)
	detector = FakeDetector(image, boxes)
	embeddings = FakeEmbeddingGenerator()
	repository = FakeRepository(people)
	matcher = FakeMatcher(results)
	workflow = ImageRecognitionWorkflow(detector, embeddings, repository, matcher)
	image_path = tmp_path / "input.png"
	image_path.write_bytes(b"test image")
	return workflow, image_path, detector, embeddings, repository, matcher


def test_invalid_input_path_is_rejected(tmp_path):
	workflow, _, *_ = make_workflow(tmp_path, [], [])

	with pytest.raises(ImageRecognitionError, match="does not exist"):
		workflow.recognize(tmp_path / "missing.png")


def test_zero_faces_returns_empty_results(tmp_path):
	workflow, image_path, _, _, repository, _ = make_workflow(tmp_path, [], [])

	report = workflow.recognize(image_path)

	assert report.faces == ()
	assert repository.calls == 1


def test_one_known_face_returns_person_result(tmp_path):
	match = RecognitionResult(True, 7, "Alice", 0.96)
	workflow, image_path, _, embeddings, _, matcher = make_workflow(
		tmp_path, [(10, 20, 30, 40)], [match], people=[object()]
	)

	report = workflow.recognize(image_path)

	assert report.faces == (FaceRecognitionResult((10, 20, 30, 40), True, 7, "Alice", 0.96),)
	assert len(embeddings.crops) == 1
	assert matcher.embeddings[0].shape == (2,)


def test_one_unknown_face_returns_unknown(tmp_path):
	workflow, image_path, *_ = make_workflow(
		tmp_path, [(1, 2, 10, 12)], [RecognitionResult(False, None, None, 0.31)], people=[object()]
	)

	face = workflow.recognize(image_path).faces[0]

	assert face.recognized is False
	assert face.person_id is None
	assert face.name is None
	assert face.similarity == pytest.approx(0.31)


def test_multiple_faces_keep_boxes_and_names(tmp_path):
	boxes = [(1, 2, 10, 12), (40, 50, 20, 30)]
	workflow, image_path, _, embeddings, _, _ = make_workflow(
		tmp_path,
		boxes,
		[RecognitionResult(True, 1, "Alice", 0.9), RecognitionResult(False, None, None, 0.2)],
		people=[object()],
	)

	report = workflow.recognize(image_path)

	assert [face.box for face in report.faces] == boxes
	assert [face.name for face in report.faces] == ["Alice", None]
	assert len(embeddings.crops) == 2


def test_results_do_not_expose_embeddings(tmp_path):
	workflow, image_path, *_ = make_workflow(
		tmp_path, [(1, 2, 10, 12)], [RecognitionResult(True, 1, "Alice", 0.9)], people=[object()]
	)

	result = workflow.recognize(image_path).faces[0]

	assert "embedding" not in result.__dict__


def test_annotation_can_be_saved(tmp_path):
	output_path = tmp_path / "nested" / "recognized.png"
	workflow, image_path, *_ = make_workflow(
		tmp_path, [(10, 10, 20, 20)], [RecognitionResult(False, None, None, 0.2)]
	)

	report = workflow.recognize(image_path, output_path=output_path)

	assert output_path.is_file()
	assert report.annotated_image is not None
	assert cv2.imread(str(output_path)) is not None


def test_empty_database_marks_face_unknown(tmp_path):
	workflow, image_path, _, _, repository, matcher = make_workflow(
		tmp_path, [(1, 2, 10, 12)], [RecognitionResult(False, None, None, None)], people=[]
	)

	face = workflow.recognize(image_path).faces[0]

	assert repository.people == []
	assert matcher.embeddings
	assert face.recognized is False
