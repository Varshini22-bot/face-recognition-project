import csv

from PIL import Image

from app.evaluation.image_inventory import inventory_images, write_inventory


def test_inventory_records_dimensions_and_readability(tmp_path):
	image = tmp_path / "face.png"
	Image.new("RGB", (13, 7), color="white").save(image)
	(tmp_path / "not_image.jpg").write_bytes(b"broken")

	records = inventory_images([tmp_path])

	assert len(records) == 2
	valid = next(record for record in records if record.filename == "face.png")
	broken = next(record for record in records if record.filename == "not_image.jpg")
	assert (valid.width, valid.height, valid.readable) == (13, 7, True)
	assert broken.readable is False


def test_inventory_writes_csv(tmp_path):
	output = tmp_path / "inventory.csv"
	write_inventory([], output)

	with output.open(newline="", encoding="utf-8") as file:
		row = next(csv.reader(file))
	assert row == ["full_path", "filename", "extension", "width", "height", "readable"]