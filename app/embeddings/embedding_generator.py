"""ArcFace embedding generation through the supported DeepFace API."""

from pathlib import Path
from typing import Any, Callable, TypeAlias

import cv2
import numpy as np


ImageInput: TypeAlias = str | Path | np.ndarray
RepresentFunction: TypeAlias = Callable[..., list[dict[str, Any]]]


class EmbeddingGenerationError(RuntimeError):
    """Base error for failures while generating a face embedding."""


class FaceNotDetectedError(EmbeddingGenerationError):
    """Raised when DeepFace cannot find a face in the input."""


class MultipleFacesDetectedError(EmbeddingGenerationError):
    """Raised when an input contains more than one face."""


class EmbeddingGenerator:
    """Generate one ArcFace embedding from an image or detected face crop."""

    MODEL_NAME = "ArcFace"
    MIN_FACE_SIZE = 20

    def __init__(
        self,
        detector_backend: str = "yunet",
        represent_function: RepresentFunction | None = None,
    ) -> None:
        """Configure ArcFace representation and the DeepFace detector backend."""

        if not detector_backend:
            raise ValueError("detector_backend must not be empty")

        self._detector_backend = detector_backend
        self._represent_function = represent_function

    def generate_embedding(
        self,
        image: ImageInput,
        detected_face: bool = False,
    ) -> np.ndarray:
        """
        Return one finite float32 ArcFace embedding.

        When detected_face=True, the supplied image is already a single
        face crop produced by the YuNet detector. DeepFace detection is
        skipped in this case.

        When detected_face=False, DeepFace performs its own face detection.
        """

        self._validate_image_input(image)

        if detected_face:
            image = self._prepare_detected_face(image)
            detector_backend = "skip"
            enforce_detection = False
        else:
            detector_backend = self._detector_backend
            enforce_detection = True

        try:
            results = self._represent(
                image,
                model_name=self.MODEL_NAME,
                enforce_detection=enforce_detection,
                detector_backend=detector_backend,
                align=True,
                normalization="ArcFace",
            )

        except Exception as error:
            error_text = str(error)

            if "face" in error_text.lower() and (
                "detect" in error_text.lower()
                or "no face" in error_text.lower()
            ):
                raise FaceNotDetectedError(
                    "No face was detected in the input image"
                ) from error

            raise EmbeddingGenerationError(
                "DeepFace could not generate an ArcFace embedding"
            ) from error

        if not results:
            raise FaceNotDetectedError(
                "No face was detected in the input image"
            )

        if len(results) != 1:
            raise MultipleFacesDetectedError(
                f"Expected one face, but detected {len(results)} faces"
            )

        try:
            embedding = np.asarray(
                results[0]["embedding"],
                dtype=np.float32,
            ).reshape(-1)

        except (KeyError, TypeError, ValueError) as error:
            raise EmbeddingGenerationError(
                "DeepFace returned an invalid embedding"
            ) from error

        if embedding.size == 0:
            raise EmbeddingGenerationError(
                "DeepFace returned an empty embedding"
            )

        if not np.isfinite(embedding).all():
            raise EmbeddingGenerationError(
                "DeepFace returned an embedding containing invalid values"
            )

        return embedding

    def _represent(
        self,
        image: ImageInput,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Call the supported DeepFace representation API."""

        if self._represent_function is not None:
            return self._represent_function(image, **kwargs)

        from deepface import DeepFace

        return DeepFace.represent(
            image,
            **kwargs,
        )

    @classmethod
    def _prepare_detected_face(
        cls,
        image: ImageInput,
    ) -> ImageInput:
        """
        Prepare a face crop that has already been detected by YuNet.

        OpenCV images are normally BGR, so convert them to RGB before
        passing them to DeepFace.
        """

        if isinstance(image, (str, Path)):
            path = Path(image)

            if not path.is_file():
                raise FileNotFoundError(
                    f"Could not read image: {path}"
                )

            return image

        if not isinstance(image, np.ndarray):
            raise ValueError(
                "image must be a non-empty image path or NumPy array"
            )

        if image.size == 0:
            raise ValueError(
                "detected face crop must not be empty"
            )

        if image.ndim not in (2, 3):
            raise ValueError(
                "detected face crop must be grayscale or a color image"
            )

        height, width = image.shape[:2]

        if width < cls.MIN_FACE_SIZE or height < cls.MIN_FACE_SIZE:
            raise FaceNotDetectedError(
                "Detected face crop is too small for ArcFace: "
                f"{width}x{height} pixels"
            )

        if image.ndim == 2:
            return cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB,
            )

        if image.shape[2] == 3:
            return cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

        if image.shape[2] == 4:
            return cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2RGB,
            )

        raise ValueError(
            "detected face crop must have 1, 3, or 4 channels"
        )

    @staticmethod
    def _validate_image_input(
        image: ImageInput,
    ) -> None:
        """Validate a path or NumPy image input."""

        if isinstance(image, (str, Path)):
            if not Path(image).is_file():
                raise FileNotFoundError(
                    f"Could not read image: {image}"
                )
            return

        if not isinstance(image, np.ndarray):
            raise ValueError(
                "image must be a non-empty image path or NumPy array"
            )

        if image.size == 0:
            raise ValueError(
                "image must be a non-empty image path or NumPy array"
            )

        if image.ndim not in (2, 3):
            raise ValueError(
                "image must be grayscale or a color image"
            )