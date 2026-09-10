from fastapi.testclient import TestClient

from api.main import app


def test_health_does_not_require_recognition_models():
	client = TestClient(app)

	response = client.get("/api/health")

	assert response.status_code == 200
	assert response.json() == {"status": "ok", "service": "VisionID API"}