---
title: VisionID ZeroGPU
a emoji: face_with_monocle
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: hf_app.py
pinned: false
---

# VisionID on Hugging Face Spaces

The Space entrypoint is `hf_app.py`. It is intentionally named differently from the `app/` package so imports such as `from app.config import AppConfig` resolve to the package rather than a root-level `app.py` module.

This Space is a Gradio adapter around the existing VisionID implementation. It preserves the required pipeline:

`OpenCV YuNet -> DeepFace ArcFace -> 512-dimensional embedding -> cosine similarity -> threshold 0.50 -> Known/Unknown`

The adapter does not implement a second matcher and does not use DeepFace verification thresholds. It calls the existing registration and recognition workflows, including SQLite storage.

## ZeroGPU compatibility

The adapter includes the current `spaces.GPU` decorator when the Space-provided `spaces` package is available. The Space runtime supplies that package, so it is intentionally not pinned in `requirements.txt`; this avoids conflicting with the Space builder's managed version. TensorFlow/DeepFace/ArcFace must still initialize successfully in the Space runtime; ZeroGPU support is not assumed. If TensorFlow cannot execute in that environment, this experiment must stop rather than replacing the pipeline with a mock or lighter model.

## Storage limitation

Free Space storage and SQLite state are suitable only for a sample/demo session. Rebuilds, lifecycle events, or a duplicated Space may reset local data. The application does not silently replace SQLite or claim production persistence.

## Required validation

The Space is not considered complete when it merely builds. Validate startup, YuNet detection, ArcFace initialization, registration, known recognition, unknown rejection, multiple-face behavior, and the Vercel frontend result display with real images.
