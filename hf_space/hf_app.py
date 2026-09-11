"""Hugging Face Gradio adapter for the existing VisionID workflows.

This module intentionally delegates all detection, embedding, matching, and
persistence to the project modules. It does not implement a second matcher.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr

try:
    import spaces
except ImportError:  # Local smoke tests do not need the ZeroGPU package.
    spaces = None

from app.config import AppConfig
from app.detection.face_detector import FaceDetector
from app.embeddings.embedding_generator import EmbeddingGenerator
from app.recognition.matcher import EmbeddingMatcher
from app.storage.database import Database
from app.storage.face_repository import FaceRepository
from app.workflows.image_recognition import ImageRecognitionWorkflow
from app.workflows.registration import RegistrationWorkflow


@lru_cache(maxsize=1)
def _services() -> tuple[RegistrationWorkflow, ImageRecognitionWorkflow]:
    config = AppConfig.from_environment()
    config.database_path.parent.mkdir(parents=True, exist_ok=True)
    config.registered_faces_dir.mkdir(parents=True, exist_ok=True)
    detector = FaceDetector()
    embedder = EmbeddingGenerator()
    repository = FaceRepository(Database(config.database_path))
    matcher = EmbeddingMatcher()
    return (
        RegistrationWorkflow(detector, embedder, repository, config),
        ImageRecognitionWorkflow(detector, embedder, repository, matcher),
    )


def _gpu_call(function):
    if spaces is None:
        return function
    return spaces.GPU(duration=60)(function)


def _copy_input(image: str | None) -> Path:
    if not image:
        raise ValueError("Upload an image first.")
    source = Path(image)
    if not source.is_file():
        raise ValueError("The uploaded image is unavailable.")
    destination = Path(tempfile.mkstemp(suffix=source.suffix or ".png")[1])
    shutil.copy2(source, destination)
    return destination


def _report_json(report) -> str:
    return json.dumps(
        {
            "faces": [
                {
                    "decision": "known" if face.recognized else "unknown",
                    "person_id": face.person_id,
                    "name": face.name,
                    "similarity": face.similarity,
                    "box": list(face.box),
                }
                for face in report.faces
            ],
            "face_count": len(report.faces),
            "threshold": 0.50,
            "embedding": "ArcFace 512-D",
            "detector": "YuNet",
        },
        indent=2,
    )


@_gpu_call
def register_person(name: str, image: str | None) -> str:
    path = _copy_input(image)
    try:
        registration, _ = _services()
        result = registration.register(name, path)
        return json.dumps(
            {"status": "registered", "id": result.person.id, "name": result.person.name},
            indent=2,
        )
    except Exception as error:
        return json.dumps({"status": "error", "message": str(error)}, indent=2)
    finally:
        path.unlink(missing_ok=True)


@_gpu_call
def recognize_image(image: str | None) -> tuple[str | None, str]:
    path = _copy_input(image)
    try:
        _, recognition = _services()
        report = recognition.recognize(path)
        annotated = recognition.annotate_image(
            __import__("cv2").imread(str(path)), report.faces, show_similarity=True
        )
        output = Path(tempfile.mkstemp(suffix=".jpg")[1])
        __import__("cv2").imwrite(str(output), annotated)
        return str(output), _report_json(report)
    except Exception as error:
        return None, json.dumps({"status": "error", "message": str(error)}, indent=2)
    finally:
        path.unlink(missing_ok=True)


def health() -> str:
    config = AppConfig.from_environment()
    return json.dumps(
        {
            "status": "ready",
            "pipeline": "YuNet -> DeepFace ArcFace -> 512-D cosine matcher",
            "threshold": 0.50,
            "storage": str(config.database_path),
            "zero_gpu_note": "ZeroGPU is optional here; TensorFlow compatibility must be verified at Space runtime.",
        },
        indent=2,
    )


with gr.Blocks(title="VisionID ZeroGPU") as demo:
    gr.Markdown("# VisionID\nYuNet + DeepFace ArcFace + cosine similarity (threshold 0.50)")
    with gr.Tab("Register"):
        name = gr.Textbox(label="Person name")
        registration_image = gr.Image(type="filepath", label="One-face image")
        register_button = gr.Button("Register")
        registration_result = gr.Code(label="Registration result", language="json")
        register_button.click(register_person, [name, registration_image], registration_result)
    with gr.Tab("Recognize"):
        recognition_image = gr.Image(type="filepath", label="Image")
        recognize_button = gr.Button("Recognize")
        annotated = gr.Image(label="Annotated result")
        recognition_result = gr.Code(label="Recognition result", language="json")
        recognize_button.click(recognize_image, recognition_image, [annotated, recognition_result])
    gr.Markdown("This is a sample/demo Space. Free Space storage is not production-persistent.")


if __name__ == "__main__":
    demo.launch()
