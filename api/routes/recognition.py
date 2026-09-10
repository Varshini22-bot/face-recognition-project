from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["recognition"])


# ============================================================================
# Configuration
# ============================================================================

# Keep these names because other API modules import them.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

ALLOWED_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}


# ============================================================================
# Lazy workflow
# ============================================================================

_workflow = None


def get_workflow():
    """
    Lazily create the image-recognition workflow.

    DeepFace/TensorFlow models are expensive to initialize, so they are
    created only when recognition is actually requested.
    """
    global _workflow

    if _workflow is None:
        from app.workflows.image_recognition import ImageRecognitionWorkflow

        _workflow = ImageRecognitionWorkflow()

    return _workflow


# ============================================================================
# Response helpers
# ============================================================================

def _success(data: dict[str, Any]) -> JSONResponse:
    """Return a successful API response."""
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "data": data,
        },
    )


def _error(
    code: str,
    message: str,
    status_code: int,
) -> JSONResponse:
    """Return a standardized API error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": {
                "code": code,
                "message": message,
            },
        },
    )


# ============================================================================
# Upload validation
# ============================================================================

async def _save_upload_temporarily(file: UploadFile) -> str:
    """
    Validate and save an uploaded image to a temporary file.

    The caller is responsible for deleting the returned temporary file.
    """

    # ------------------------------------------------------------------------
    # Validate MIME type
    # ------------------------------------------------------------------------

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise ValueError(
            "Unsupported image type. "
            "Please upload a JPG, PNG, or WebP image."
        )

    # ------------------------------------------------------------------------
    # Read file
    # ------------------------------------------------------------------------

    data = await file.read()

    if not data:
        raise ValueError("The uploaded image is empty.")

    # ------------------------------------------------------------------------
    # Validate file size
    # ------------------------------------------------------------------------

    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(
            "The uploaded image is too large. "
            "Maximum size is 10 MB."
        )

    # ------------------------------------------------------------------------
    # Determine extension
    # ------------------------------------------------------------------------

    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in ALLOWED_SUFFIXES:
        suffix = ".jpg"

    # ------------------------------------------------------------------------
    # Create temporary file
    # ------------------------------------------------------------------------

    temporary_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    try:
        temporary_file.write(data)
        temporary_file.flush()
    finally:
        temporary_file.close()

    return temporary_file.name


# ============================================================================
# Recognition endpoint
# ============================================================================

@router.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    """
    Recognize faces in an uploaded image.

    Internal exception details are logged server-side but are never returned
    to the frontend.
    """

    temporary_path: str | None = None

    try:
        # ====================================================================
        # 1. Validate and save upload
        # ====================================================================

        try:
            temporary_path = await _save_upload_temporarily(file)

        except ValueError as error:
            return _error(
                "INVALID_IMAGE",
                str(error),
                400,
            )

        # ====================================================================
        # 2. Get recognition workflow
        # ====================================================================

        workflow = get_workflow()

        # ====================================================================
        # 3. Run recognition
        # ====================================================================

        report = workflow.recognize(temporary_path)

        if report is None:
            return _error(
                "PROCESSING_ERROR",
                "Unable to process this image.",
                500,
            )

        # ====================================================================
        # 4. Convert result to JSON-safe data
        # ====================================================================

        if hasattr(report, "to_dict"):
            result = report.to_dict()

        elif isinstance(report, dict):
            result = report

        else:
            result = {}

            for attribute in (
                "faces",
                "face_count",
                "processing_time_ms",
            ):
                if hasattr(report, attribute):
                    result[attribute] = getattr(report, attribute)

        # ====================================================================
        # 5. Return result
        # ====================================================================

        return _success(result)

    # =========================================================================
    # IMPORTANT SECURITY HANDLING
    # =========================================================================

    except Exception:
        """
        Log the real exception and traceback on the server.

        Do NOT expose the internal exception message to the browser.

        This prevents messages such as:

            RuntimeError: internal detail

        from being returned through the public API.
        """

        logger.exception("Recognition request failed")

        return _error(
            "PROCESSING_ERROR",
            "Unable to process this image.",
            500,
        )

    # =========================================================================
    # Cleanup
    # =========================================================================

    finally:

        if temporary_path:
            try:
                os.remove(temporary_path)

            except OSError:
                logger.warning(
                    "Could not delete temporary upload: %s",
                    temporary_path,
                )

        try:
            await file.close()

        except Exception:
            logger.warning(
                "Could not close uploaded file",
                exc_info=True,
            )