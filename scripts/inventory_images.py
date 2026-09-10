"""Inventory candidate image files for manual evaluation-set selection."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.image_inventory import inventory_images, write_inventory


def main() -> int:
	parser = argparse.ArgumentParser(description="Inventory image metadata without face recognition.")
	parser.add_argument("--source", type=Path, action="append", required=True, help="Folder to scan; repeat for multiple folders")
	parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "evaluation" / "image_inventory.csv")
	args = parser.parse_args()
	records = inventory_images(args.source)
	write_inventory(records, args.output)
	print(f"Inventoried {len(records)} supported image files")
	print(f"Inventory saved to: {args.output}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())