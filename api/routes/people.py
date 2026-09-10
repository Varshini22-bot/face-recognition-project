"""Registered-people management API routes."""

import io
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from api.routes.recognition import (
	ALLOWED_CONTENT_TYPES,
	ALLOWED_SUFFIXES,
	MAX_UPLOAD_BYTES,
)
from api.schemas import DeleteResponse, ErrorResponse, PeopleResponse, PersonResponse, RegistrationResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def get_registration_workflow():
	"""Create registration dependencies lazily, after health/docs startup."""
	from app.config import AppConfig
	from app.detection.face_detector import FaceDetector
	from app.embeddings.embedding_generator import EmbeddingGenerator
	from app.storage.database import Database
	from app.storage.face_repository import FaceRepository
	from app.workflows.registration import RegistrationWorkflow

	config = AppConfig.from_environment()
	return RegistrationWorkflow(
		FaceDetector(),
		EmbeddingGenerator(),
		FaceRepository(Database(config.database_path)),
		config,
	)


def get_repository():
	from app.config import AppConfig
	from app.storage.database import Database
	from app.storage.face_repository import FaceRepository

	config = AppConfig.from_environment()
	return FaceRepository(Database(config.database_path))


def _error(code: str, message: str, status_code: int) -> JSONResponse:
	return JSONResponse(
		status_code=status_code,
		content=ErrorResponse(error={"code": code, "message": message}).model_dump(),
	)


def _person_response(person) -> PersonResponse:
	return PersonResponse(id=person.id, name=person.name, created_at=person.created_at)


@router.get("/people", response_model=PeopleResponse)
def people() -> PeopleResponse | JSONResponse:
	try:
		registered = get_repository().get_all_people()
		return PeopleResponse(
			people=[_person_response(person) for person in registered],
			count=len(registered),
		)
	except Exception:
		logger.exception("Could not load registered people")
		return _error("DATABASE_ERROR", "Registered people could not be loaded.", 500)


@router.post("/people/register", response_model=RegistrationResponse, responses={400: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def register_person(
	name: str | None = Form(default=None),
	file: UploadFile | None = File(default=None),
) -> RegistrationResponse | JSONResponse:
	if not name or not name.strip():
		return _error("EMPTY_NAME", "Person name is required.", 400)
	if file is None:
		return _error("MISSING_FILE", "Please upload a reference image.", 400)
	if file.content_type not in ALLOWED_CONTENT_TYPES:
		return _error("INVALID_IMAGE", "The uploaded file is not a supported image type.", 400)

	suffix = Path(file.filename or "upload.png").suffix.lower()
	if suffix not in ALLOWED_SUFFIXES:
		suffix = ".png"
	temporary_path: Path | None = None
	try:
		content = await file.read(MAX_UPLOAD_BYTES + 1)
		if not content:
			return _error("EMPTY_UPLOAD", "The uploaded image is empty.", 400)
		if len(content) > MAX_UPLOAD_BYTES:
			return _error("FILE_TOO_LARGE", "The uploaded image exceeds the 10 MB limit.", 413)
		try:
			with Image.open(io.BytesIO(content)) as image:
				image.verify()
		except (OSError, UnidentifiedImageError):
			return _error("INVALID_IMAGE", "The uploaded file is not a valid image.", 400)

		with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
			temporary.write(content)
			temporary_path = Path(temporary.name)
		try:
			result = get_registration_workflow().register(name, temporary_path)
		except Exception as error:
			message = str(error)
			if "already registered" in message.lower():
				return _error("DUPLICATE_PERSON", "That person name is already registered.", 409)
			if "no face" in message.lower():
				return _error("NO_FACE", "Registration requires exactly one visible face.", 400)
			if "multiple" in message.lower() or "exactly one" in message.lower():
				return _error("MULTIPLE_FACES", "Registration requires exactly one visible face.", 400)
			if "image" in message.lower() or "read" in message.lower():
				return _error("INVALID_IMAGE", "The uploaded image could not be processed.", 400)
			raise
		return RegistrationResponse(person=_person_response(result.person))
	except Exception:
		logger.exception("Registration request failed")
		return _error("REGISTRATION_ERROR", "Unable to register this person.", 500)
	finally:
		if temporary_path is not None:
			temporary_path.unlink(missing_ok=True)


@router.delete("/people/{person_id}", response_model=DeleteResponse, responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
def delete_person(person_id: int) -> DeleteResponse | JSONResponse:
	try:
		from app.storage.face_repository import PersonNotFoundError
		get_repository().delete_person(person_id)
		return DeleteResponse(deleted_id=person_id)
	except PersonNotFoundError:
		return _error("PERSON_NOT_FOUND", "That registered person does not exist.", 404)
	except Exception:
		logger.exception("Could not delete registered person")
		return _error("DATABASE_ERROR", "The person could not be deleted.", 500)