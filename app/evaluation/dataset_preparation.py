"""Prepare manually labeled images for evaluation without using recognition."""

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image, UnidentifiedImageError


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class DatasetPreparationError(ValueError):
	"""Raised when evaluation dataset preparation input is invalid."""


class FaceValidationDetector(Protocol):
	def detect_image(self, image_path: Path): ...


@dataclass(frozen=True)
class FaceValidation:
	"""Face-count check for one manually selected image."""

	image_path: Path
	face_count: int | None
	valid: bool
	warning: str | None


@dataclass(frozen=True)
class PreparationSummary:
	"""Summary of files copied into one labeled dataset group."""

	kind: str
	label: str | None
	added: int
	destination: Path
	warnings: tuple[str, ...] = ()


def prepare_dataset(
	source_dir: str | Path,
	dataset_dir: str | Path,
	kind: str,
	person_name: str | None = None,
	selected_files: list[str | Path] | None = None,
	face_detector: FaceValidationDetector | None = None,
) -> PreparationSummary:
	"""Copy readable source images into a manually selected known/unknown group.

	When ``selected_files`` is provided, only those explicit paths are copied.
	"""
	source = Path(source_dir)
	dataset = Path(dataset_dir)
	_normalize_kind(kind)
	label = _clean_person_name(person_name) if kind == "known" else None
	if kind == "known" and label is None:
		raise DatasetPreparationError("person_name is required for known images")
	if not source.is_dir():
		raise DatasetPreparationError(f"Source directory does not exist: {source}")

	destination = dataset / kind / label if kind == "known" else dataset / "unknown"
	destination.mkdir(parents=True, exist_ok=True)
	added = 0
	warnings: list[str] = []
	if selected_files is None:
		files = sorted(path for path in source.rglob("*") if path.is_file())
	else:
		files = [Path(path) for path in selected_files]
	for source_file in files:
		if not source_file.is_absolute():
			source_file = source / source_file
		if not source_file.is_file():
			raise DatasetPreparationError(f"Selected image does not exist: {source_file}")
		if source_file.suffix.lower() not in SUPPORTED_SUFFIXES or not _is_readable_image(source_file):
			if selected_files is not None:
				raise DatasetPreparationError(f"Selected file is not a readable supported image: {source_file}")
			continue
		if face_detector is not None:
			validation = validate_image_faces(source_file, face_detector)
			if validation.warning is not None:
				warnings.append(validation.warning)
			if kind == "known" and not validation.valid:
				continue
		shutil.copy2(source_file, _unique_destination(destination, source_file.name))
		added += 1
	return PreparationSummary(kind, label, added, destination, tuple(warnings))


def validate_image_faces(
	image_path: str | Path,
	detector: FaceValidationDetector,
) -> FaceValidation:
	"""Validate that an image contains exactly one detectable face."""
	path = Path(image_path)
	try:
		_, faces = detector.detect_image(path)
	except Exception as error:
		return FaceValidation(path, None, False, f"Could not validate {path.name}: {error}")
	count = len(faces)
	if count == 1:
		return FaceValidation(path, count, True, None)
	return FaceValidation(
		path,
		count,
		False,
		f"{path.name}: detected {count} faces; known evaluation images require exactly one face",
	)


def _normalize_kind(kind: str) -> None:
	if kind not in {"known", "unknown"}:
		raise DatasetPreparationError("kind must be either 'known' or 'unknown'")


def _clean_person_name(person_name: str | None) -> str | None:
	if person_name is None:
		return None
	cleaned = re.sub(r"[^A-Za-z0-9 _-]+", "", person_name).strip()
	cleaned = re.sub(r"\s+", "_", cleaned)
	return cleaned or None


def _is_readable_image(path: Path) -> bool:
	try:
		with Image.open(path) as image:
			image.verify()
		return True
	except (OSError, UnidentifiedImageError):
		return False


def _unique_destination(directory: Path, filename: str) -> Path:
	base = Path(filename).stem
	suffix = Path(filename).suffix.lower()
	candidate = directory / f"{base}{suffix}"
	index = 1
	while candidate.exists():
		candidate = directory / f"{base}_{index}{suffix}"
		index += 1
	return candidate