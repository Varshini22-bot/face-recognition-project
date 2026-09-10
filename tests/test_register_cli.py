import importlib.util
from pathlib import Path

import numpy as np

from app.storage.face_repository import PersonRecord


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "register_person.py"
SPEC = importlib.util.spec_from_file_location("register_person_cli", SCRIPT_PATH)
register_person = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(register_person)


def fake_result():
	return type(
		"RegistrationResult",
		(),
		{
			"person": PersonRecord(
				7, "Varshini", Path("data/registered_faces/Varshini.png"), np.ones(512), "now"
			)
		},
	)()


def test_cli_registers_with_required_arguments(capsys):
	class FakeWorkflow:
		def register(self, name, image):
			assert name == "Varshini"
			assert image == Path("face.jpg")
			return fake_result()

	def factory(database_path):
		assert database_path is None
		return FakeWorkflow()

	assert register_person.main(["--name", "Varshini", "--image", "face.jpg"], factory) == 0
	output = capsys.readouterr().out
	assert "Varshini" in output
	assert "database ID 7" in output


def test_cli_passes_optional_database_path(capsys):
	class FakeWorkflow:
		def register(self, name, image):
			return fake_result()

	seen = []

	def factory(database_path):
		seen.append(database_path)
		return FakeWorkflow()

	assert register_person.main(
		["--name", "Varshini", "--image", "face.jpg", "--db", "temp.db"], factory
	) == 0
	assert seen == [Path("temp.db")]


def test_cli_reports_expected_registration_errors(capsys):
	class FakeWorkflow:
		def register(self, name, image):
			raise register_person.RegistrationError("No face detected")

	assert register_person.main(
		["--name", "Varshini", "--image", "missing.jpg"], lambda path: FakeWorkflow()
	) == 1
	assert "No face detected" in capsys.readouterr().err