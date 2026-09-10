"""Register one person's face from a command-line image path."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from app.config import AppConfig
from app.detection.face_detector import FaceDetector
from app.embeddings.embedding_generator import EmbeddingGenerator
from app.storage.database import Database
from app.storage.face_repository import FaceRepository
from app.workflows.registration import RegistrationError, RegistrationWorkflow


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	"""Parse registration CLI arguments."""
	parser = argparse.ArgumentParser(description="Register one person's face.")
	parser.add_argument("--name", required=True, help="Person's name")
	parser.add_argument("--image", required=True, type=Path, help="Path to a face image")
	parser.add_argument(
		"--db",
		type=Path,
		help="Optional SQLite database path; otherwise FACE_RECOGNITION_DB_PATH is used",
	)
	return parser.parse_args(argv)


def create_workflow(database_path: Path | None = None) -> RegistrationWorkflow:
	"""Build the existing registration workflow without duplicating its logic."""
	config = AppConfig.from_environment()
	if database_path is not None:
		config = AppConfig(database_path, config.registered_faces_dir)
	return RegistrationWorkflow(
		FaceDetector(),
		EmbeddingGenerator(),
		FaceRepository(Database(config.database_path)),
		config,
	)


def main(argv: list[str] | None = None, workflow_factory=create_workflow) -> int:
	"""Register a person and return a process exit code."""
	args = parse_args(argv)
	try:
		result = workflow_factory(args.db).register(args.name, args.image)
	except RegistrationError as error:
		print(f"Registration failed: {error}", file=sys.stderr)
		return 1
	except (OSError, ValueError, RuntimeError) as error:
		print(f"Registration failed: {error}", file=sys.stderr)
		return 1

	print(
		f"Successfully registered '{result.person.name}' "
		f"with database ID {result.person.id}."
	)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
