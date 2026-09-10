import json

import pytest

from app.evaluation.evaluator import (
	EvaluationReport,
	EvaluationRunner,
	EvaluationSample,
	Prediction,
	calculate_counts,
	calculate_metrics,
	load_dataset,
	write_report,
)
from app.recognition.matcher import RecognitionResult
from app.workflows.image_recognition import FaceRecognitionResult, ImageRecognitionReport


def predictions():
	return [
		Prediction("Alice", "Alice", 0.9, "alice.png"),
		Prediction("Bob", None, 0.4, "bob.png"),
		Prediction(None, "Alice", 0.8, "unknown.png"),
		Prediction(None, None, None, "unknown2.png"),
	]


def test_counts_and_metrics():
	counts = calculate_counts(predictions(), 0.5)
	metrics = calculate_metrics(predictions(), 0.5)

	assert counts.tp == 1
	assert counts.tn == 1
	assert counts.fp == 1
	assert counts.fn == 1
	assert metrics.accuracy == pytest.approx(0.5)
	assert metrics.precision == pytest.approx(0.5)
	assert metrics.recall == pytest.approx(0.5)
	assert metrics.f1 == pytest.approx(0.5)
	assert metrics.far == pytest.approx(0.5)
	assert metrics.frr == pytest.approx(0.5)


def test_threshold_sweep_and_report_without_embeddings(tmp_path):
	class FakeWorkflow:
		def recognize(self, image_path):
			name = "Alice" if image_path.name == "alice.png" else None
			return ImageRecognitionReport(
				image_path,
				(FaceRecognitionResult((0, 0, 1, 1), name is not None, 1 if name else None, name, 0.8),),
			)

	report = EvaluationRunner(lambda: FakeWorkflow()).evaluate(
		[EvaluationSample(tmp_path / "alice.png", "Alice"), EvaluationSample(tmp_path / "unknown.png", None)],
		threshold=0.5,
		thresholds=[0.5, 0.9],
	)
	output = tmp_path / "report.json"
	write_report(report, output)
	data = json.loads(output.read_text(encoding="utf-8"))

	assert len(report.threshold_sweep) == 2
	assert report.best_threshold_on_dataset == pytest.approx(0.5)
	assert report.metrics.tp == 1
	assert report.diagnostics[0].image_filename == "alice.png"
	assert report.diagnostics[0].expected_identity == "Alice"
	assert report.diagnostics[0].detected_face_count == 1
	assert report.diagnostics[0].predicted_identity == "Alice"
	assert report.diagnostics[0].similarity == pytest.approx(0.8)
	assert report.diagnostics[0].accepted is True
	assert report.diagnostics[0].threshold_used == pytest.approx(0.5)
	assert report.diagnostics[1].predicted_identity is None
	assert report.diagnostics[1].accepted is False
	assert "embedding" not in output.read_text(encoding="utf-8")
	assert "embedding" not in data


def test_empty_dataset_is_rejected():
	with pytest.raises(ValueError, match="No labeled"):
		EvaluationRunner(lambda: None).evaluate([])


def test_dataset_loader_supports_known_and_unknown(tmp_path):
	(tmp_path / "known" / "Alice").mkdir(parents=True)
	(tmp_path / "unknown").mkdir()
	(tmp_path / "known" / "Alice" / "a.png").write_bytes(b"x")
	(tmp_path / "unknown" / "u.jpg").write_bytes(b"x")

	samples = load_dataset(tmp_path)

	assert [(sample.ground_truth, sample.image_path.name) for sample in samples] == [
		("Alice", "a.png"),
		(None, "u.jpg"),
	]