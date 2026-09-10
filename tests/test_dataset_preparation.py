from pathlib import Path

from PIL import Image

from app.evaluation.dataset_preparation import (
	DatasetPreparationError,
	prepare_dataset,
	validate_image_faces,
)


def write_image(path: Path) -> None:
	Image.new("RGB", (8, 8), color="white").save(path)


def test_known_images_are_copied_and_originals_remain(tmp_path):
	source = tmp_path / "source"
	source.mkdir()
	write_image(source / "person.jpg")
	dataset = tmp_path / "evaluation"

	summary = prepare_dataset(source, dataset, "known", "Varshini Test")

	assert summary.added == 1
	assert summary.destination == dataset / "known" / "Varshini_Test"
	assert (summary.destination / "person.jpg").is_file()
	assert (source / "person.jpg").is_file()


def test_unknown_images_are_copied_without_person_label(tmp_path):
	source = tmp_path / "source"
	source.mkdir()
	write_image(source / "unknown.png")

	summary = prepare_dataset(source, tmp_path / "evaluation", "unknown")

	assert summary.added == 1
	assert summary.label is None
	assert (tmp_path / "evaluation" / "unknown" / "unknown.png").is_file()


def test_duplicate_filenames_are_not_overwritten(tmp_path):
	source = tmp_path / "source"
	source.mkdir()
	write_image(source / "face.jpg")
	destination = tmp_path / "evaluation"
	prepare_dataset(source, destination, "unknown")

	nested_source = source / "nested"
	nested_source.mkdir()
	write_image(nested_source / "face.jpg")
	prepare_dataset(source, destination, "unknown")

	assert (destination / "unknown" / "face.jpg").is_file()
	assert (destination / "unknown" / "face_1.jpg").is_file()


def test_unreadable_and_unsupported_files_are_skipped(tmp_path):
	source = tmp_path / "source"
	source.mkdir()
	(source / "notes.txt").write_text("not an image", encoding="utf-8")
	(source / "broken.png").write_bytes(b"not an image")

	summary = prepare_dataset(source, tmp_path / "evaluation", "unknown")

	assert summary.added == 0


def test_known_name_and_input_validation(tmp_path):
	source = tmp_path / "source"
	source.mkdir()

	try:
		prepare_dataset(source, tmp_path / "evaluation", "known")
		raise AssertionError("known registration should require a person name")
	except DatasetPreparationError as error:
		assert "person_name" in str(error)

	try:
		prepare_dataset(source, tmp_path / "evaluation", "other", "Name")
		raise AssertionError("invalid kind should be rejected")
	except DatasetPreparationError as error:
		assert "known" in str(error)


def test_explicit_files_copy_only_selected_images(tmp_path):
	source = tmp_path / "source"
	source.mkdir()
	write_image(source / "selected.png")
	write_image(source / "not_selected.png")

	summary = prepare_dataset(
		source, tmp_path / "evaluation", "known", "Varshini", [source / "selected.png"]
	)

	assert summary.added == 1
	assert (summary.destination / "selected.png").is_file()
	assert not (summary.destination / "not_selected.png").exists()


class FakeDetector:
	def __init__(self, face_count):
		self.face_count = face_count

	def detect_image(self, image_path):
		return None, [(0, 0, 1, 1)] * self.face_count


def test_known_image_with_exactly_one_face_is_copied(tmp_path):
	source = tmp_path / "source"
	source.mkdir()
	write_image(source / "one.jpg")

	summary = prepare_dataset(
		source, tmp_path / "evaluation", "known", "Varshini", face_detector=FakeDetector(1)
	)

	assert summary.added == 1
	assert summary.warnings == ()


def test_known_zero_or_multiple_faces_are_reported_and_not_copied(tmp_path):
	source = tmp_path / "source"
	source.mkdir()
	write_image(source / "zero.jpg")
	write_image(source / "many.jpg")

	summary = prepare_dataset(
		source, tmp_path / "evaluation", "known", "Varshini", face_detector=FakeDetector(0)
	)

	assert summary.added == 0
	assert len(summary.warnings) == 2
	assert "detected 0 faces" in summary.warnings[0]

	summary = prepare_dataset(
		source, tmp_path / "evaluation2", "known", "Varshini", face_detector=FakeDetector(2)
	)
	assert summary.added == 0
	assert all("detected 2 faces" in warning for warning in summary.warnings)


def test_unknown_image_validation_warns_but_preserves_manual_label(tmp_path):
	source = tmp_path / "source"
	source.mkdir()
	write_image(source / "unknown.jpg")

	summary = prepare_dataset(
		source, tmp_path / "evaluation", "unknown", face_detector=FakeDetector(2)
	)

	assert summary.added == 1
	assert len(summary.warnings) == 1


def test_validate_image_faces_returns_exact_count(tmp_path):
	image = tmp_path / "image.jpg"
	write_image(image)

	validation = validate_image_faces(image, FakeDetector(1))

	assert validation.face_count == 1
	assert validation.valid is True