import sqlite3

import numpy as np
import pytest

from app.config import AppConfig
from app.storage.database import Database
from app.storage.face_repository import (
	DuplicatePersonError,
	FaceRepository,
	PersonNotFoundError,
)
from app.workflows.registration import (
	InvalidRegistrationImageError,
	MultipleFacesDetectedError,
	NoFaceDetectedError,
	RegistrationError,
	RegistrationWorkflow,
)


class FakeDetector:
	def __init__(self, faces):
		self.faces = faces

	def detect_image(self, image_path):
		return np.zeros((100, 100, 3), dtype=np.uint8), self.faces


class FakeEmbeddingGenerator:
	def generate_embedding(self, image, detected_face=False):
		assert detected_face is True
		return np.arange(512, dtype=np.float32)


@pytest.fixture
def setup_registration(tmp_path):
	config = AppConfig(tmp_path / "db" / "faces.db", tmp_path / "registered")
	repository = FaceRepository(Database(config.database_path))
	return config, repository


def make_image(tmp_path):
	image = tmp_path / "input.png"
	image.write_bytes(b"test image")
	return image


def make_workflow(config, repository, faces):
	return RegistrationWorkflow(
		FakeDetector(faces), FakeEmbeddingGenerator(), repository, config
	)


def test_successful_registration_and_image_path(setup_registration, tmp_path):
	config, repository = setup_registration
	result = make_workflow(config, repository, [(10, 10, 20, 20)]).register(
		"Ada Lovelace", make_image(tmp_path)
	)

	assert result.person.name == "Ada Lovelace"
	assert result.person.embedding.shape == (512,)
	assert result.person.image_path.parent == config.registered_faces_dir
	assert result.person.image_path.is_file()


def test_invalid_image_is_rejected(setup_registration, tmp_path):
	config, repository = setup_registration

	with pytest.raises(InvalidRegistrationImageError):
		make_workflow(config, repository, [(1, 1, 5, 5)]).register("Ada", tmp_path / "missing.png")


def test_no_face_is_rejected(setup_registration, tmp_path):
	config, repository = setup_registration

	with pytest.raises(NoFaceDetectedError):
		make_workflow(config, repository, []).register("Ada", make_image(tmp_path))


def test_multiple_faces_are_rejected(setup_registration, tmp_path):
	config, repository = setup_registration

	with pytest.raises(MultipleFacesDetectedError, match="exactly one face"):
		make_workflow(config, repository, [(1, 1, 5, 5), (10, 10, 5, 5)]).register(
			"Ada", make_image(tmp_path)
		)


def test_duplicate_name_is_rejected(setup_registration, tmp_path):
	config, repository = setup_registration
	image = make_image(tmp_path)
	workflow = make_workflow(config, repository, [(1, 1, 5, 5)])
	workflow.register("Ada", image)

	with pytest.raises(RegistrationError, match="already registered"):
		workflow.register("ada", image)


def test_repository_retrieval_and_deletion(setup_registration):
	_, repository = setup_registration
	created = repository.create_person("Grace", "data/registered_faces/grace.png", np.ones(512))

	assert repository.get_person_by_id(created.id).name == "Grace"
	assert repository.get_person_by_name("grace").id == created.id
	assert repository.get_all_people()[0].id == created.id
	repository.delete_person(created.id)

	with pytest.raises(PersonNotFoundError):
		repository.get_person_by_id(created.id)


def test_duplicate_repository_name_is_rejected(setup_registration):
	_, repository = setup_registration
	repository.create_person("Grace", "grace.png", np.ones(3))

	with pytest.raises(DuplicatePersonError):
		repository.create_person("grace", "other.png", np.ones(3))


def test_embedding_serialization_round_trip(setup_registration):
	_, repository = setup_registration
	embedding = np.arange(5, dtype=np.float32)
	created = repository.create_person("Lin", "lin.png", embedding)
	retrieved = repository.get_person_by_id(created.id)

	assert retrieved.embedding.dtype == np.float32
	assert np.array_equal(retrieved.embedding, embedding)


def test_database_uses_parameterized_storage(setup_registration):
	config, repository = setup_registration
	repository.create_person("O'Connor", "oconnor.png", np.ones(2))
	with sqlite3.connect(config.database_path) as connection:
		row = connection.execute("SELECT embedding FROM persons").fetchone()
	assert isinstance(row[0], bytes)
