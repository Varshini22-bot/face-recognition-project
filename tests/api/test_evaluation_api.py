import json

from fastapi.testclient import TestClient

from api.main import app
from api.routes import evaluation


def test_no_report_returns_empty_state(monkeypatch, tmp_path):
	monkeypatch.setattr(evaluation, "_report_path", lambda: tmp_path / "missing.json")
	response = TestClient(app).get("/api/evaluation")
	assert response.status_code == 200
	assert response.json() == {"available": False, "message": "No evaluation report available."}


def test_valid_report_maps_real_metrics_and_hides_paths(tmp_path, monkeypatch):
	report = {
		"number_of_images": 2,
		"number_of_faces_evaluated": 2,
		"number_of_known_samples": 1,
		"number_of_unknown_samples": 1,
		"threshold_used": 0.5,
		"best_threshold_on_dataset": 0.3,
		"metrics": {"accuracy": 0.8, "precision": 0.9, "recall": 0.7, "f1": 0.8, "far": 0.1, "frr": 0.2},
		"threshold_sweep": [{"threshold": 0.3, "f1": 1.0}],
		"diagnostics": [{"image_filename": "sample.jpg", "expected_identity": "Varshini", "detected_face_count": 1, "predicted_identity": "Varshini", "similarity": 0.8, "accepted": True, "threshold_used": 0.5, "image_path": "C:\\private\\face.jpg"}],
	}
	path = tmp_path / "report.json"
	path.write_text(json.dumps(report), encoding="utf-8")
	monkeypatch.setattr(evaluation, "_report_path", lambda: path)

	response = TestClient(app).get("/api/evaluation")
	data = response.json()
	assert response.status_code == 200
	assert data["available"] is True
	assert data["accuracy"] == 0.8
	assert data["best_threshold"] == 0.3
	assert data["model"] == "ArcFace"
	assert data["detector"] == "YuNet"
	assert data["diagnostics"][0]["image_filename"] == "sample.jpg"
	assert "image_path" not in response.text
	assert "embedding" not in response.text


def test_malformed_report_returns_unavailable_state(tmp_path, monkeypatch):
	path = tmp_path / "report.json"
	path.write_text("not json", encoding="utf-8")
	monkeypatch.setattr(evaluation, "_report_path", lambda: path)

	response = TestClient(app).get("/api/evaluation")

	assert response.status_code == 200
	assert response.json()["available"] is False