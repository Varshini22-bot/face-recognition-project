from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from api.routes import people


class FakePerson:
	def __init__(self, person_id=1, name="Varshini"):
		self.id = person_id
		self.name = name
		self.created_at = "2026-09-10T12:00:00+00:00"


class FakeRepository:
	def __init__(self, records=None):
		self.records = records or []
		self.deleted = []

	def get_all_people(self):
		return self.records

	def delete_person(self, person_id):
		from app.storage.face_repository import PersonNotFoundError
		for person in self.records:
			if person.id == person_id:
				self.deleted.append(person_id)
				return
		raise PersonNotFoundError("not found")


class FakeRegistrationWorkflow:
	def __init__(self, result=None, error=None):
		self.result = result or type("Result", (), {"person": FakePerson()})()
		self.error = error
		self.paths = []

	def register(self, name, path):
		self.paths.append(Path(path))
		if self.error:
			raise self.error
		return self.result


def valid_png():
	buffer = BytesIO()
	Image.new("RGB", (2, 2), "white").save(buffer, format="PNG")
	return buffer.getvalue()


def test_get_people_empty(monkeypatch):
	monkeypatch.setattr(people, "get_repository", lambda: FakeRepository())
	response = TestClient(app).get("/api/people")
	assert response.status_code == 200
	assert response.json() == {"people": [], "count": 0}


def test_get_people_hides_embeddings(monkeypatch):
	person = FakePerson()
	person.embedding = [1.0, 2.0]
	monkeypatch.setattr(people, "get_repository", lambda: FakeRepository([person]))
	response = TestClient(app).get("/api/people")
	assert response.status_code == 200
	assert response.json() == {"people": [{"id": 1, "name": "Varshini", "created_at": person.created_at}], "count": 1}
	assert "embedding" not in response.text


def test_register_valid_person_and_cleanup(monkeypatch):
	workflow = FakeRegistrationWorkflow()
	monkeypatch.setattr(people, "get_registration_workflow", lambda: workflow)
	response = TestClient(app).post(
		"/api/people/register",
		data={"name": "Varshini"},
		files={"file": ("reference.png", valid_png(), "image/png")},
	)
	assert response.status_code == 200
	assert response.json()["person"]["name"] == "Varshini"
	assert workflow.paths and not workflow.paths[0].exists()


def test_register_empty_name_and_invalid_image():
	client = TestClient(app)
	empty_name = client.post("/api/people/register", data={"name": " "}, files={"file": ("x.png", valid_png(), "image/png")})
	invalid = client.post("/api/people/register", data={"name": "A"}, files={"file": ("x.txt", b"bad", "text/plain")})
	assert empty_name.status_code == 400
	assert empty_name.json()["error"]["code"] == "EMPTY_NAME"
	assert invalid.status_code == 400
	assert invalid.json()["error"]["code"] == "INVALID_IMAGE"


def test_register_face_count_and_duplicate_errors(monkeypatch):
	client = TestClient(app)
	for error, code in [
		(RuntimeError("No face detected; registration requires exactly one face"), "NO_FACE"),
		(RuntimeError("Detected 2 faces; registration requires exactly one face"), "MULTIPLE_FACES"),
		(RuntimeError("A person named 'Varshini' is already registered"), "DUPLICATE_PERSON"),
	]:
		monkeypatch.setattr(people, "get_registration_workflow", lambda error=error: FakeRegistrationWorkflow(error=error))
		response = client.post("/api/people/register", data={"name": "Varshini"}, files={"file": ("x.png", valid_png(), "image/png")})
		assert response.status_code == (409 if code == "DUPLICATE_PERSON" else 400)
		assert response.json()["error"]["code"] == code


def test_delete_existing_and_missing_person(monkeypatch):
	repository = FakeRepository([FakePerson(3, "Grace")])
	monkeypatch.setattr(people, "get_repository", lambda: repository)
	client = TestClient(app)
	deleted = client.delete("/api/people/3")
	missing = client.delete("/api/people/9")
	assert deleted.status_code == 200
	assert deleted.json() == {"success": True, "deleted_id": 3}
	assert missing.status_code == 404
	assert missing.json()["error"]["code"] == "PERSON_NOT_FOUND"