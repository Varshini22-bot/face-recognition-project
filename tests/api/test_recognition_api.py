from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app
from api.routes import recognition


class FakeFace:
	def __init__(self, recognized, name, similarity):
		self.recognized = recognized
		self.name = name
		self.similarity = similarity


class FakeMatcher:
	class Threshold:
		value = 0.5

	_threshold = Threshold()


class FakeWorkflow:
	def __init__(self, faces):
		self.faces = faces
		self.paths = []

	def recognize(self, path):
		self.paths.append(Path(path))
		return type("Report", (), {"faces": self.faces})()


def png_bytes():
	return b"\x89PNG\r\n\x1a\n" + b"not-a-real-png"


def test_missing_upload():
	response = TestClient(app).post("/api/recognize")
	assert response.status_code == 400
	assert response.json()["error"]["code"] == "MISSING_FILE"


def test_invalid_and_empty_uploads():
	client = TestClient(app)
	invalid = client.post("/api/recognize", files={"file": ("x.txt", b"hello", "text/plain")})
	empty = client.post("/api/recognize", files={"file": ("x.png", b"", "image/png")})
	assert invalid.status_code == 400
	assert invalid.json()["error"]["code"] == "INVALID_IMAGE"
	assert empty.status_code == 400
	assert empty.json()["error"]["code"] == "EMPTY_UPLOAD"


def test_known_unknown_multiple_and_cleanup(monkeypatch, tmp_path):
	workflow = FakeWorkflow([FakeFace(True, "Varshini", 0.83), FakeFace(False, None, 0.31)])
	workflow._matcher = FakeMatcher()
	monkeypatch.setattr(recognition, "get_workflow", lambda: workflow)
	client = TestClient(app)

	# Use a real valid tiny image for the workflow contract and cleanup assertion.
	from PIL import Image
	from io import BytesIO
	buffer = BytesIO()
	Image.new("RGB", (2, 2), "white").save(buffer, format="PNG")
	response = client.post("/api/recognize", files={"file": ("face.png", buffer.getvalue(), "image/png")})
	assert response.status_code == 200
	assert response.json()["faces"][0]["status"] == "known"
	assert response.json()["faces"][1]["status"] == "unknown"
	assert all(not path.exists() for path in workflow.paths)


def test_unexpected_workflow_error(monkeypatch):
	class BrokenWorkflow:
		def recognize(self, path):
			raise RuntimeError("internal detail")

	monkeypatch.setattr(recognition, "get_workflow", lambda: BrokenWorkflow())
	from PIL import Image
	from io import BytesIO
	buffer = BytesIO()
	Image.new("RGB", (2, 2), "white").save(buffer, format="PNG")
	response = TestClient(app).post("/api/recognize", files={"file": ("face.png", buffer.getvalue(), "image/png")})
	assert response.status_code == 500
	assert response.json()["error"]["code"] == "PROCESSING_ERROR"
	assert "internal detail" not in response.text