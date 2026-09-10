"""Read-only evaluation report API."""

import json
import logging
from pathlib import Path

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter()
REPORT_PATH = Path(__file__).resolve().parents[2] / "data" / "evaluation" / "evaluation_report.json"


def _report_path() -> Path:
	return Path(__import__("os").environ.get("VISIONID_EVALUATION_REPORT", REPORT_PATH))


@router.get("/evaluation")
def evaluation() -> dict:
	"""Return the existing evaluation report without running inference."""
	path = _report_path()
	if not path.is_file():
		return {"available": False, "message": "No evaluation report available."}
	try:
		report = json.loads(path.read_text(encoding="utf-8"))
		metrics = report["metrics"]
		response = {
			"available": True,
			"threshold": report["threshold_used"],
			"best_threshold": report["best_threshold_on_dataset"],
			"accuracy": metrics["accuracy"],
			"precision": metrics["precision"],
			"recall": metrics["recall"],
			"f1": metrics["f1"],
			"far": metrics["far"],
			"frr": metrics["frr"],
			"known_samples": report["number_of_known_samples"],
			"unknown_samples": report["number_of_unknown_samples"],
			"images_evaluated": report["number_of_images"],
			"faces_evaluated": report["number_of_faces_evaluated"],
			"model": "ArcFace",
			"detector": "YuNet",
			"threshold_sweep": report.get("threshold_sweep", []),
			"diagnostics": [
				{
					"image_filename": item.get("image_filename"),
					"expected_identity": item.get("expected_identity"),
					"detected_face_count": item.get("detected_face_count"),
					"predicted_identity": item.get("predicted_identity"),
					"similarity": item.get("similarity"),
					"accepted": item.get("accepted"),
				}
				for item in report.get("diagnostics", [])
			],
		}
		return response
	except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
		logger.exception("Evaluation report is malformed")
		return {"available": False, "message": "Evaluation report is unavailable or malformed."}