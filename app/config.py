"""Configurable paths for local face registration data."""

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AppConfig:
	"""Filesystem locations used by the application."""

	database_path: Path
	registered_faces_dir: Path

	@classmethod
	def from_environment(cls) -> "AppConfig":
		"""Build configuration from environment variables and project defaults."""
		data_dir = PROJECT_ROOT / "data"
		return cls(
			database_path=Path(
				os.environ.get("FACE_RECOGNITION_DB_PATH", data_dir / "face_recognition.db")
			),
			registered_faces_dir=Path(
				os.environ.get("REGISTERED_FACES_DIR", data_dir / "registered_faces")
			),
		)
