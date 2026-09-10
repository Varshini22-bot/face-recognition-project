"""Create a metadata inventory of candidate images without recognition."""

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.evaluation.dataset_preparation import SUPPORTED_SUFFIXES


@dataclass(frozen=True)
class ImageInventoryRecord:
	"""Non-biometric file and image metadata."""

	full_path: str
	filename: str
	extension: str
	width: int | None
	height: int | None
	readable: bool


def inventory_images(source_dirs: list[str | Path]) -> list[ImageInventoryRecord]:
	"""Inventory supported image files recursively under the supplied folders."""
	records: list[ImageInventoryRecord] = []
	for source_dir in source_dirs:
		root = Path(source_dir)
		if not root.is_dir():
			continue
		for path in sorted(root.rglob("*")):
			if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
				records.append(_inspect_image(path))
	return records


def write_inventory(records: list[ImageInventoryRecord], output_path: str | Path) -> None:
	"""Write inventory records as CSV."""
	output = Path(output_path)
	output.parent.mkdir(parents=True, exist_ok=True)
	with output.open("w", newline="", encoding="utf-8") as file:
		writer = csv.DictWriter(file, fieldnames=list(asdict(records[0]).keys()) if records else [
			"full_path", "filename", "extension", "width", "height", "readable"
		])
		writer.writeheader()
		writer.writerows(asdict(record) for record in records)


def _inspect_image(path: Path) -> ImageInventoryRecord:
	width = height = None
	readable = False
	try:
		with Image.open(path) as image:
			width, height = image.size
			image.verify()
		readable = True
	except (OSError, UnidentifiedImageError):
		pass
	return ImageInventoryRecord(str(path.resolve()), path.name, path.suffix.lower(), width, height, readable)