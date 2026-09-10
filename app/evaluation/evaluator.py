"""Metrics and dataset evaluation for the existing image-recognition pipeline."""

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from app.recognition.matcher import EmbeddingMatcher
from app.recognition.threshold import SimilarityThreshold
from app.workflows.image_recognition import ImageRecognitionReport, ImageRecognitionWorkflow


UNKNOWN_LABEL = "Unknown"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class EvaluationSample:
	"""One image and its expected identity, or ``None`` for an unknown face."""

	image_path: Path
	ground_truth: str | None


@dataclass(frozen=True)
class Prediction:
	"""A privacy-safe prediction used for metric calculation."""

	ground_truth: str | None
	predicted_name: str | None
	similarity: float | None
	image_path: Path


@dataclass(frozen=True)
class ConfusionCounts:
	"""Binary counts where positive means correct known-identity recognition."""

	tp: int
	tn: int
	fp: int
	fn: int


@dataclass(frozen=True)
class Metrics:
	"""Classification and acceptance metrics for one threshold."""

	threshold: float
	accuracy: float
	precision: float
	recall: float
	f1: float
	far: float
	frr: float
	tp: int
	tn: int
	fp: int
	fn: int


@dataclass(frozen=True)
class EvaluationDiagnostic:
	"""Per-image/face diagnostic data without embeddings."""

	image_filename: str
	expected_identity: str | None
	detected_face_count: int
	predicted_identity: str | None
	similarity: float | None
	accepted: bool
	threshold_used: float


@dataclass(frozen=True)
class EvaluationReport:
	"""Serializable evaluation summary without biometric vectors."""

	number_of_images: int
	number_of_faces_evaluated: int
	number_of_known_samples: int
	number_of_unknown_samples: int
	threshold_used: float
	best_threshold_on_dataset: float
	metrics: Metrics
	threshold_sweep: tuple[Metrics, ...]
	average_recognition_time_seconds: float
	total_processing_time_seconds: float
	average_time_per_image_seconds: float
	confusion_matrix: dict[str, dict[str, int]] | None
	diagnostics: tuple[EvaluationDiagnostic, ...]

	def to_dict(self) -> dict:
		"""Return JSON-safe report data."""
		return asdict(self)


def calculate_counts(predictions: Iterable[Prediction], threshold: float) -> ConfusionCounts:
	"""Calculate counts for exact known identity and unknown rejection.

	Positive means a known sample was assigned its correct identity. Therefore,
	known-but-wrong and known-as-unknown are false negatives; an unknown sample
	assigned any identity is a false positive. This definition makes FAR and FRR
	interpretable for the open-set recognition behavior of this project.
	"""
	tp = tn = fp = fn = 0
	for prediction in predictions:
		accepted = (
			prediction.predicted_name is not None
			and prediction.similarity is not None
			and prediction.similarity >= threshold
		)
		if prediction.ground_truth is None:
			if accepted:
				fp += 1
			else:
				tn += 1
		elif accepted and prediction.predicted_name == prediction.ground_truth:
			tp += 1
		else:
			fn += 1
	return ConfusionCounts(tp, tn, fp, fn)


def calculate_metrics(predictions: Iterable[Prediction], threshold: float) -> Metrics:
	"""Calculate accuracy, precision, recall, F1, FAR, and FRR safely."""
	counts = calculate_counts(predictions, threshold)
	total = counts.tp + counts.tn + counts.fp + counts.fn
	accuracy = _safe_ratio(counts.tp + counts.tn, total)
	precision = _safe_ratio(counts.tp, counts.tp + counts.fp)
	recall = _safe_ratio(counts.tp, counts.tp + counts.fn)
	f1 = _safe_ratio(2 * precision * recall, precision + recall)
	far = _safe_ratio(counts.fp, counts.fp + counts.tn)
	frr = _safe_ratio(counts.fn, counts.fn + counts.tp)
	return Metrics(
		threshold, accuracy, precision, recall, f1, far, frr,
		counts.tp, counts.tn, counts.fp, counts.fn,
	)


def load_dataset(dataset_path: str | Path) -> list[EvaluationSample]:
	"""Load ``known/<person>/*`` and ``unknown/*`` samples."""
	root = Path(dataset_path)
	if not root.is_dir():
		raise FileNotFoundError(f"Evaluation dataset directory does not exist: {root}")
	samples: list[EvaluationSample] = []
	known_dir = root / "known"
	if known_dir.is_dir():
		for person_dir in sorted(path for path in known_dir.iterdir() if path.is_dir()):
			for image in _images_under(person_dir):
				samples.append(EvaluationSample(image, person_dir.name))
	unknown_dir = root / "unknown"
	if unknown_dir.is_dir():
		for image in _images_under(unknown_dir):
			samples.append(EvaluationSample(image, None))
	return samples


class EvaluationRunner:
	"""Evaluate images through ``ImageRecognitionWorkflow`` without changing it."""

	def __init__(self, workflow_factory: Callable[[], ImageRecognitionWorkflow]) -> None:
		self._workflow_factory = workflow_factory

	def evaluate(
		self,
		samples: Sequence[EvaluationSample],
		threshold: float = 0.5,
		thresholds: Sequence[float] | None = None,
	) -> EvaluationReport:
		"""Evaluate labeled samples and return metrics plus a threshold sweep."""
		if not samples:
			raise ValueError("No labeled evaluation images were found")
		if not -1.0 <= threshold <= 1.0:
			raise ValueError("threshold must be between -1 and 1")
		sweep_values = tuple(thresholds or (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90))
		if any(value < -1.0 or value > 1.0 for value in sweep_values):
			raise ValueError("threshold sweep values must be between -1 and 1")

		workflow = self._workflow_factory()
		predictions: list[Prediction] = []
		total_time = 0.0
		recognition_time = 0.0
		faces_evaluated = 0
		diagnostics: list[EvaluationDiagnostic] = []
		for sample in samples:
			started = time.perf_counter()
			report = workflow.recognize(sample.image_path)
			elapsed = time.perf_counter() - started
			total_time += elapsed
			recognition_time += elapsed
			if not report.faces:
				predictions.append(Prediction(sample.ground_truth, None, None, sample.image_path))
				diagnostics.append(
					EvaluationDiagnostic(
						sample.image_path.name, sample.ground_truth, 0, None, None, False, threshold
					)
				)
				continue
			faces_evaluated += len(report.faces)
			for face in report.faces:
				predictions.append(
					Prediction(sample.ground_truth, face.name, face.similarity, sample.image_path)
				)
				diagnostics.append(
					EvaluationDiagnostic(
						sample.image_path.name,
						sample.ground_truth,
						len(report.faces),
						face.name,
						face.similarity,
						face.name is not None
						and face.similarity is not None
						and face.similarity >= threshold,
						threshold,
					)
				)

		metrics = calculate_metrics(predictions, threshold)
		sweep = tuple(calculate_metrics(predictions, value) for value in sweep_values)
		best_threshold = max(sweep, key=lambda item: (item.f1, item.accuracy, -item.threshold)).threshold
		return EvaluationReport(
			number_of_images=len(samples),
			number_of_faces_evaluated=faces_evaluated,
			number_of_known_samples=sum(sample.ground_truth is not None for sample in samples),
			number_of_unknown_samples=sum(sample.ground_truth is None for sample in samples),
			threshold_used=threshold,
			best_threshold_on_dataset=best_threshold,
			metrics=metrics,
			threshold_sweep=sweep,
			average_recognition_time_seconds=recognition_time / faces_evaluated if faces_evaluated else 0.0,
			total_processing_time_seconds=total_time,
			average_time_per_image_seconds=total_time / len(samples),
			confusion_matrix=_build_confusion_matrix(predictions, threshold),
			diagnostics=tuple(diagnostics),
		)


def write_report(report: EvaluationReport, output_path: str | Path) -> None:
	"""Write the JSON evaluation report."""
	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)
	output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def write_sweep_csv(report: EvaluationReport, output_path: str | Path) -> None:
	"""Write threshold-sweep metrics as CSV."""
	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)
	with output.open("w", newline="", encoding="utf-8") as file:
		writer = csv.DictWriter(file, fieldnames=list(asdict(report.metrics).keys()))
		writer.writeheader()
		writer.writerows(asdict(metric) for metric in report.threshold_sweep)


def _images_under(directory: Path) -> list[Path]:
	return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _safe_ratio(numerator: float, denominator: float) -> float:
	return numerator / denominator if denominator else 0.0


def _build_confusion_matrix(
	predictions: Iterable[Prediction], threshold: float
) -> dict[str, dict[str, int]] | None:
	known_labels = sorted({prediction.ground_truth for prediction in predictions if prediction.ground_truth})
	if len(known_labels) < 2:
		return None
	labels = [*known_labels, UNKNOWN_LABEL]
	matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}
	for prediction in predictions:
		actual = prediction.ground_truth or UNKNOWN_LABEL
		accepted = prediction.predicted_name is not None and prediction.similarity is not None and prediction.similarity >= threshold
		predicted = prediction.predicted_name if accepted and prediction.predicted_name in labels else UNKNOWN_LABEL
		matrix[actual][predicted] += 1
	return matrix