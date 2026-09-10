"""Image recognition API route."""

import logging
import io
import os
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from api.schemas import ErrorResponse, FaceResponse, RecognitionResponse

logger = logging.getLogger(__name__)
router = APIRouter()
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp", "image/tiff"}
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def get_workflow():
	"""Create the local workflow lazily so health/docs do not load recognition models."""
	from app.config import AppConfig
	from app.detection.face_detector import FaceDetector
	from app.embeddings.embedding_generator import EmbeddingGenerator
	from app.recognition.matcher import EmbeddingMatcher
	from app.storage.database import Database
	from app.storage.face_repository import FaceRepository
	from app.workflows.image_recognition import ImageRecognitionWorkflow

	config = AppConfig.from_environment()
	return ImageRecognitionWorkflow(
		FaceDetector(),
		EmbeddingGenerator(),
		FaceRepository(Database(config.database_path)),
		EmbeddingMatcher(),
	)


def _error(code: str, message: str, status_code: int) -> JSONResponse:
	return JSONResponse(
		status_code=status_code,
		content=ErrorResponse(error={"code": code, "message": message}).model_dump(),
	)


@router.post("/recognize", response_model=RecognitionResponse, responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def recognize(file: UploadFile | None = File(default=None)) -> RecognitionResponse | JSONResponse:
	"""Recognize faces in a temporary uploaded image."""
	if file is None:
		return _error("MISSING_FILE", "Please upload an image file.", 400)
	if file.content_type not in ALLOWED_CONTENT_TYPES:
		return _error("INVALID_IMAGE", "The uploaded file is not a supported image type.", 400)

	suffix = Path(file.filename or "upload.png").suffix.lower()
	if suffix not in ALLOWED_SUFFIXES:
		suffix = ".png"
	temporary_path: Path | None = None
	try:
		content = await file.read(MAX_UPLOAD_BYTES + 1)
		if not content:
			return _error("EMPTY_UPLOAD", "The uploaded file is empty.", 400)
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
		started = time.perf_counter()
		workflow = get_workflow()
		report = workflow.recognize(temporary_path)
		elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
		from app.recognition.threshold import SimilarityThreshold
		threshold = SimilarityThreshold().value
		faces = [
			FaceResponse(
				recognized=face.recognized,
				name=face.name,
				similarity=face.similarity,
				threshold=threshold,
				status="known" if face.recognized else "unknown",
			)
			for face in report.faces
		]
		return RecognitionResponse(faces=faces, face_count=len(faces), processing_time_ms=elapsed_ms)
	except Exception:
		logger.exception("Recognition request failed")
		return _error("PROCESSING_ERROR", "Unable to process this image.", 500)
	finally:
		if temporary_path is not None:
			temporary_path.unlink(missing_ok=True)