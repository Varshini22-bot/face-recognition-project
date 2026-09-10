"""Copy manually labeled images into the evaluation dataset structure."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from app.detection.face_detector import FaceDetector
from app.evaluation.dataset_preparation import DatasetPreparationError, prepare_dataset


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Prepare manually labeled images for face-recognition evaluation."
	)
	parser.add_argument("--source", type=Path, help="Folder containing source images")
	parser.add_argument(
		"--file",
		type=Path,
		action="append",
		dest="selected_files",
		help="Explicit image path to copy; repeat for each selected image",
	)
	parser.add_argument("--kind", choices=("known", "unknown"), help="Image label group")
	parser.add_argument("--person-name", help="Known person's label")
	parser.add_argument(
		"--validate-faces",
		action="store_true",
		help="Use YuNet to report zero/multiple faces; known invalid images are not copied",
	)
	parser.add_argument(
		"--dataset",
		type=Path,
		default=PROJECT_ROOT / "data" / "evaluation",
		help="Evaluation dataset root",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	source = args.source or Path(input("Source image folder: ").strip())
	kind = args.kind or input("Images are for known or unknown? [known/unknown]: ").strip().lower()
	person_name = args.person_name
	if kind == "known" and person_name is None:
		person_name = input("Known person's name: ").strip()

	try:
		detector = FaceDetector() if args.validate_faces else None
		summary = prepare_dataset(
			source, args.dataset, kind, person_name, args.selected_files, detector
		)
	except DatasetPreparationError as error:
		print(f"Error: {error}", file=sys.stderr)
		return 1

	if summary.kind == "known":
		print(f"Added {summary.added} images for known person '{summary.label}'")
	else:
		print(f"Added {summary.added} images to unknown dataset")
	print(f"Destination: {summary.destination}")
	for warning in summary.warnings:
		print(f"Warning: {warning}", file=sys.stderr)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())