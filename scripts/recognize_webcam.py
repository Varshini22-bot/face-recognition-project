"""Run real-time face recognition from a webcam."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from app.config import AppConfig
from app.detection.face_detector import FaceDetector
from app.embeddings.embedding_generator import EmbeddingGenerator
from app.recognition.matcher import EmbeddingMatcher
from app.storage.database import Database
from app.storage.face_repository import FaceRepository
from app.workflows.webcam_recognition import (
	WebcamRecognitionError,
	WebcamRecognitionWorkflow,
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Recognize faces from a webcam.")
	parser.add_argument(
		"--camera",
		type=int,
		default=0,
		help="Camera index to use (default: 0)",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	try:
		config = AppConfig.from_environment()
		workflow = WebcamRecognitionWorkflow(
			FaceDetector(),
			EmbeddingGenerator(),
			FaceRepository(Database(config.database_path)),
			EmbeddingMatcher(),
		)
		report = workflow.run(camera_index=args.camera)
	except WebcamRecognitionError as error:
		print(f"Error: {error}", file=sys.stderr)
		return 1

	print(f"Processed frames: {report.frames_processed}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
