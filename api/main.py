"""FastAPI application for local VisionID integration."""

import os

# Keep TensorFlow's thread pools bounded before route imports initialize DeepFace.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.people import router as people_router
from api.routes.recognition import router as recognition_router
from api.routes.evaluation import router as evaluation_router
from app.config import AppConfig


def _cors_origins() -> list[str]:
    """
    Read allowed frontend origins from the environment.

    For local development, allow the Vite development server.
    For production, configure the deployed frontend origin with:

        VISIONID_CORS_ORIGINS=https://visionid-murex.vercel.app
    """
    value = os.environ.get(
        "VISIONID_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,https://visionid-murex.vercel.app",
    )

    return [
        origin.strip().rstrip("/")
        for origin in value.split(",")
        if origin.strip()
    ]


app = FastAPI(
    title="VisionID API",
    version="0.1.0",
)


# ============================================================================
# CORS
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=r"^https:\/\/visionid-murex[a-z0-9-]*\.vercel\.app$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ============================================================================
# Health endpoint
# ============================================================================

@app.get("/api/health")
def health() -> dict[str, str]:
    """Return a lightweight process health response."""
    return {
        "status": "ok",
        "service": "VisionID API",
    }


@app.get("/api/ready")
def readiness() -> dict[str, object]:
    """Report whether the configured local data paths are ready for requests."""
    config = AppConfig.from_environment()
    config.database_path.parent.mkdir(parents=True, exist_ok=True)
    config.registered_faces_dir.mkdir(parents=True, exist_ok=True)
    return {
        "status": "ready",
        "service": "VisionID API",
        "storage": {
            "database": "neon",
            "registered_faces_dir": str(config.registered_faces_dir),
        },
    }


# ============================================================================
# API routers
# ============================================================================
#
# Exposes:
#     /api/health
#     /api/recognize
#     /api/people
#     /api/people/register
#     /api/people/{person_id}
#     /api/evaluation
#

app.include_router(recognition_router, prefix="/api")
app.include_router(people_router, prefix="/api")
app.include_router(evaluation_router, prefix="/api")
