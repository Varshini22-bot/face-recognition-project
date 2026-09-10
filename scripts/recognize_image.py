"""Recognize faces in a still image and optionally save annotations."""

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from app.config import AppConfig
from app.detection.face_detector import FaceDetector
from app.embeddings.embedding_generator import EmbeddingGenerator
from app.recognition.matcher import EmbeddingMatcher
from app.storage.database import Database
from app.storage.face_repository import FaceRepository
from app.workflows.image_recognition import ImageRecognitionReport, ImageRecognitionWorkflow


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Recognize faces in an image.")
	parser.add_argument("image", type=Path, help="Path to the input image")
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		help="Path for the annotated image; omit to skip saving",
	)
	parser.add_argument(
		"--display",
		action="store_true",
		help="Display the annotated image until a key is pressed",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()

	try:
		config = AppConfig.from_environment()
		workflow = ImageRecognitionWorkflow(
			FaceDetector(),
			EmbeddingGenerator(),
			FaceRepository(Database(config.database_path)),
			EmbeddingMatcher(),
		)
		report = workflow.recognize(args.image, args.output)
	except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
		print(f"Error: {error}", file=sys.stderr)
		return 1

	print(f"Detected faces: {len(report.faces)}")
	for index, face in enumerate(report.faces, start=1):
		label = face.name if face.recognized else "Unknown"
		score = f", similarity: {face.similarity:.3f}" if face.similarity is not None else ""
		print(f"Face {index}: {label}{score}")
	if args.output:
		print(f"Annotated image saved to: {args.output}")

	if args.display:
		if report.annotated_image is None:
			report = ImageRecognitionReport(
				report.image_path,
				report.faces,
				workflow.annotate_image(
					cv2.imread(str(args.image)), report.faces
				),
			)
		cv2.imshow("Face Recognition", report.annotated_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
