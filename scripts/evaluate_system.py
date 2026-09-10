"""Evaluate the face-recognition system on a labeled local dataset."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from app.config import AppConfig
from app.detection.face_detector import FaceDetector
from app.embeddings.embedding_generator import EmbeddingGenerator
from app.evaluation.evaluator import (
	EvaluationRunner,
	load_dataset,
	write_report,
	write_sweep_csv,
)
from app.recognition.matcher import EmbeddingMatcher
from app.recognition.threshold import SimilarityThreshold
from app.storage.database import Database
from app.storage.face_repository import FaceRepository
from app.workflows.image_recognition import ImageRecognitionWorkflow


def main() -> int:
	parser = argparse.ArgumentParser(description="Evaluate face recognition on labeled images.")
	parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "evaluation")
	parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "evaluation" / "evaluation_report.json")
	parser.add_argument("--threshold", type=float, default=0.5)
	args = parser.parse_args()

	try:
		samples = load_dataset(args.dataset)
		config = AppConfig.from_environment()
		def workflow_factory():
			return ImageRecognitionWorkflow(
				FaceDetector(),
				EmbeddingGenerator(),
				FaceRepository(Database(config.database_path)),
				EmbeddingMatcher(SimilarityThreshold(-1.0)),
			)
		report = EvaluationRunner(workflow_factory).evaluate(samples, threshold=args.threshold)
		write_report(report, args.output)
		write_sweep_csv(report, args.output.with_suffix(".csv"))
	except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
		print(f"Evaluation unavailable: {error}", file=sys.stderr)
		print("Use data/evaluation/known/<person>/ and data/evaluation/unknown/ with labeled images.", file=sys.stderr)
		return 1

	print(f"Images evaluated: {report.number_of_images}")
	print(f"Faces evaluated: {report.number_of_faces_evaluated}")
	print(f"Threshold: {report.threshold_used:.2f}")
	print(f"Best threshold on this dataset: {report.best_threshold_on_dataset:.2f}")
	print(f"Accuracy: {report.metrics.accuracy:.3f}")
	print(f"FAR: {report.metrics.far:.3f}")
	print(f"FRR: {report.metrics.frr:.3f}")
	print(f"Report saved to: {args.output}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
