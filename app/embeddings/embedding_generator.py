"""ArcFace embedding generation through ONNX Runtime."""

from pathlib import Path
from typing import TypeAlias
from urllib.request import urlopen

import cv2
import numpy as np


ImageInput: TypeAlias = str | Path | np.ndarray
LandmarksInput: TypeAlias = np.ndarray | list[tuple[float, float]] | None

MODEL_NAME = "arcfaceresnet100-8.onnx"
MODEL_URL = (
    "https://huggingface.co/onnxmodelzoo/arcfaceresnet100-8/resolve/main/"
    f"{MODEL_NAME}"
)
MODEL_SHA256 = "f3a6bc281e72f88862f5748b53be3d76b3b48f8f1ab1f4a537941bdc4e1b01da"


class EmbeddingGenerationError(RuntimeError):
    """Base error for failures while generating a face embedding."""


class FaceNotDetectedError(EmbeddingGenerationError):
    """Raised when no face crop is supplied."""


class MultipleFacesDetectedError(EmbeddingGenerationError):
    """Raised when an input contains more than one face."""


class EmbeddingGenerator:
    """Generate one normalized 512-D ArcFace embedding from a face crop."""

    INPUT_SIZE = (112, 112)
    EMBEDDING_SIZE = 512
    MIN_FACE_SIZE = 20

    def __init__(self, model_path: str | Path | None = None) -> None:
        """Load the Apache-2.0 ONNX Model Zoo ArcFace model."""
        self._model_path = Path(model_path) if model_path else self._default_model_path()
        self._ensure_model(self._model_path)
        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                str(self._model_path), providers=["CPUExecutionProvider"]
            )
        except Exception as error:
            raise EmbeddingGenerationError(
                f"Could not load ArcFace ONNX model: {self._model_path}"
            ) from error

        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if not inputs or inputs[0].shape != [1, 3, 112, 112]:
            raise EmbeddingGenerationError("ArcFace model has an unexpected input contract")
        if not outputs or outputs[0].shape != [1, 512]:
            raise EmbeddingGenerationError("ArcFace model has an unexpected output contract")
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name

    def generate_embedding(
        self,
        image: ImageInput,
        detected_face: bool = False,
        landmarks: LandmarksInput = None,
    ) -> np.ndarray:
        """Return one finite, L2-normalized ArcFace embedding."""

        self._validate_image_input(image)
        if not detected_face:
            raise FaceNotDetectedError("EmbeddingGenerator requires a detected single-face crop")
        crop = self._prepare_detected_face(image)
        if landmarks is not None:
            crop = self._align_face(crop, landmarks)
        resized = cv2.resize(crop, self.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        # The ONNX Model Zoo ArcFace preprocessing uses BGR pixels and (x - 127.5) / 128.

        tensor = resized.astype(np.float32).transpose(2, 0, 1)[None, ...]
        tensor = (tensor - 127.5) / 128.0
        try:
            result = self._session.run([self._output_name], {self._input_name: tensor})[0]
        except Exception as error:
            raise EmbeddingGenerationError("ArcFace ONNX inference failed") from error
        embedding = np.asarray(result, dtype=np.float32).reshape(-1)
        if embedding.size != self.EMBEDDING_SIZE or not np.isfinite(embedding).all():
            raise EmbeddingGenerationError("ArcFace returned an invalid embedding")
        norm = float(np.linalg.norm(embedding))
        if norm <= 0.0:
            raise EmbeddingGenerationError("ArcFace returned a zero embedding")
        return embedding / norm

    @classmethod
    def _default_model_path(cls) -> Path:
        project_model = Path(__file__).resolve().parents[2] / "models" / MODEL_NAME
        if project_model.is_file():
            return project_model
        return Path("/tmp/visionid/models") / MODEL_NAME

    @staticmethod
    def _ensure_model(model_path: Path) -> None:
        if model_path.is_file():
            return
        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urlopen(MODEL_URL, timeout=120) as response:
                model_path.write_bytes(response.read())
        except Exception as error:
            model_path.unlink(missing_ok=True)
            raise OSError(f"Could not download ArcFace model from {MODEL_URL}") from error

    @classmethod
    def _prepare_detected_face(cls, image: ImageInput) -> np.ndarray:
        if isinstance(image, (str, Path)):
            image_array = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if image_array is None:
                raise FileNotFoundError(f"Could not read image: {image}")
            image = image_array
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("detected face crop must be a non-empty NumPy array")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("detected face crop must have three color channels")
        height, width = image.shape[:2]
        if width < cls.MIN_FACE_SIZE or height < cls.MIN_FACE_SIZE:
            raise FaceNotDetectedError(
                f"Detected face crop is too small for ArcFace: {width}x{height} pixels"
            )
        return image

    @staticmethod
    def _align_face(image: np.ndarray, landmarks: LandmarksInput) -> np.ndarray:
        points = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
        if points.shape != (5, 2) or not np.isfinite(points).all():
            raise ValueError("ArcFace alignment requires five finite facial landmarks")
        target = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
             [41.5493, 92.3655], [70.7299, 92.2041]],
            dtype=np.float32,
        )
        matrix, _ = cv2.estimateAffinePartial2D(points, target, method=cv2.LMEDS)
        if matrix is None:
            raise ValueError("Could not estimate ArcFace alignment transform")
        return cv2.warpAffine(image, matrix, (112, 112), borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def _validate_image_input(image: ImageInput) -> None:


        if isinstance(image, (str, Path)):
            if not Path(image).is_file():
                raise FileNotFoundError(f"Could not read image: {image}")
            return
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("image must be a non-empty image path or NumPy array")
