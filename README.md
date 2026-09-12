# VisionID — Face Recognition Identification System

VisionID is a robust, explainable facial recognition and identity management system built with Python, FastAPI, OpenCV YuNet, ONNX Runtime ArcFace, and React. The system handles enrollment, facial embedding generation, vector matching, known/unknown identity decisions, and evaluation reporting.

---

## 1. Project Overview

VisionID provides an end-to-end computer vision workspace that:
- **Enrolls Individuals**: Registers new people using a reference image containing exactly one face.
- **Generates Face Embeddings**: Extracts 512-dimensional, L2-normalized identity vectors using ArcFace.
- **Stores Identity Embeddings**: Persists registered identities and vector embeddings in a local SQLite database.
- **Identifies Faces in New Images**: Detects multiple faces in arbitrary images and matches them against registered candidates.
- **Supports Multiple Faces**: Detects and evaluates each face independently within multi-person scenes.
- **Rejects Unknown Faces**: Enforces a cosine-similarity threshold to reject unregistered individuals or low-confidence matches.

---

## 2. Assignment Context

This system was designed and developed for the **Code Nimbus AI/ML Intern** technical assessment. It satisfies all core requirements:
- **Face Detection**: Fast, robust face localization via OpenCV YuNet.
- **Face Representation**: High-accuracy 512-dimensional embeddings using the Apache-2.0 ONNX Model Zoo ArcFace model.
- **Similarity Matching**: Normalized cosine similarity ranking.
- **Unknown Face Rejection**: Strict threshold-based gating to prevent forced false positives.
- **Evaluation Framework**: Offline evaluation pipeline computing Accuracy, Precision, Recall, F1, FAR, and FRR.
- **API & UI**: Clean FastAPI REST backend with a modern React dashboard.
- **Comprehensive Documentation**: Architectural rationale, metrics, failure modes, and deployment guide.

---

## 3. Key Features

- **YuNet Face Detection**: Detects frontal and angled faces with high efficiency and low compute overhead.
- **ArcFace 512-D Embeddings**: Uses additive angular margin loss features for high inter-class separation.
- **Cosine Similarity Matching**: Vectorized cosine similarity computation across stored candidates.
- **Open-Set Rejection**: Configurable similarity threshold rejecting unknown intruders.
- **Multi-Face Analysis**: Detects, crops, embeds, and annotates multiple faces in a single frame.
- **Interactive Webcam Workflow**: Real-time webcam inference and annotation through CLI and browser interface.
- **Production REST API**: FastAPI backend with typed Pydantic schemas, standard error codes, and CORS support.
- **Modern React Dashboard**: Vite-powered responsive user interface with live pipeline visualization and identity roster.
- **Automated Evaluation Suite**: Automated dataset evaluation tool with threshold sweeps and diagnostic reports.
- **Lightweight SQLite Storage**: Self-contained database storing identity records and embedding BLOBs.

---

## 4. System Architecture

```text
Input Image / Webcam Stream
           │
           ▼
  OpenCV YuNet Face Detector (ONNX)
           │
           ▼
    Face Bounding Box & Crop
           │
           ▼
    ONNX Runtime ArcFace Model
           │
           ▼
   512-D Normalized Embedding
           │
           ▼
  SQLite Embedding Storage (BLOB)
           │
           ▼
   Cosine Similarity Matcher
           │
           ▼
   Similarity Threshold Decision (Default: 0.50)
           │
     ┌─────┴─────┐
     ▼           ▼
   Known      Unknown
  (Identity) (Rejected)
```

---

## 5. AI/ML Models

| Component | Technology | Model / Artifact | Embedding Size | Function |
|---|---|---|---|---|
| **Face Detection** | OpenCV DNN | `face_detection_yunet_2023mar.onnx` | N/A | Localizes face bounding boxes `(x, y, w, h)` |
| **Face Representation** | ONNX Runtime | `arcfaceresnet100-8.onnx` | 512 float32 | Extracts discriminative identity representations |
| **Similarity Metric** | NumPy | Cosine Similarity | Scalar $[-1, 1]$ | Measures angular vector alignment |

> **Note on Model Selection**: The project intentionally uses **YuNet** and the Apache-2.0 `arcfaceresnet100-8.onnx` ArcFace model from the official ONNX Model Zoo successor at https://huggingface.co/onnxmodelzoo/arcfaceresnet100-8. The verified model contract is input `data` `[1, 3, 112, 112]` and output `fc1` `[1, 512]`; SHA-256 is `f3a6bc281e72f88862f5748b53be3d76b3b48f8f1ab1f4a537941bdc4e1b01da`. The codebase does **not** use Haar Cascades, MTCNN, FaceNet, or landmark-mesh models, keeping inference fast and dependencies lean.

---

## 6. Recognition Logic

For each uploaded image or video frame:
1. **Detect Faces**: The image is passed to YuNet (`cv2.FaceDetectorYN`) to detect all visible face boxes.
2. **Crop Faces**: Each detected box is validated and cropped from the image.
3. **Generate ArcFace Embedding**: Each YuNet face crop is resized to 112×112, normalized with `(pixel - 127.5) / 128.0`, passed through ONNX Runtime using the verified BGR NCHW contract, and L2-normalized into a 512-dimensional float32 vector.
4. **Compare Candidates**: The query vector is compared against every registered identity vector stored in SQLite using cosine similarity:
   $$\text{similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$
5. **Rank & Select**: The candidate with the highest similarity score is selected.
6. **Threshold Comparison**: The top similarity score is compared against the configured threshold (default: `0.50`).
7. **Decision**:
   - If $\text{similarity} \ge 0.50$, classify as **`known`** with the candidate's name and similarity score.
   - If $\text{similarity} < 0.50$ (or if no candidates exist), classify as **`unknown`**.

---

## 7. Registration Process

Registration requires **exactly one visible face** to guarantee reference integrity:
- **Validation**:
  - If **zero faces** are detected $\rightarrow$ returns `NO_FACE` (HTTP 400).
  - If **multiple faces** are detected $\rightarrow$ returns `MULTIPLE_FACES` (HTTP 400).
  - If the name is empty or whitespace $\rightarrow$ returns `EMPTY_NAME` (HTTP 400).
  - If the image file is corrupt or invalid $\rightarrow$ returns `INVALID_IMAGE` (HTTP 400).
- **Duplicate Prevention**: If a person with the requested name already exists in the database $\rightarrow$ returns `DUPLICATE_PERSON` (HTTP 409 Conflict).
- **Storage**: Upon successful validation, the 512-D float32 embedding is stored in SQLite alongside a reference image saved to `data/registered_faces/`.

---

## 8. Unknown Face Rejection

In real-world recognition systems, open-set classification is essential. Without threshold-based rejection, any random person or background artifact would be forced into the closest registered identity (false acceptance).

VisionID enforces a strict cosine similarity cutoff ($0.50$ default):
- Only matches exceeding the threshold receive a positive identification.
- Any match below the threshold is safely marked as **unknown**, preventing false positive matches.

---

## 9. Evaluation Results

The evaluation suite was executed against a curated local test dataset (`scripts/evaluate_system.py --dataset data/evaluation`):

| Metric | Result (Threshold = 0.50) | Optimal Threshold (0.30) |
|---|---|---|
| **Images Evaluated** | 6 | 6 |
| **Faces Evaluated** | 6 | 6 |
| **Known Samples** | 3 | 3 |
| **Unknown Samples** | 3 | 3 |
| **Accuracy** | **Not treated as a production benchmark** | **Not treated as a production benchmark** |
| **Precision** | **Not treated as a production benchmark** | **Not treated as a production benchmark** |
| **Recall** | **Not treated as a production benchmark** | **Not treated as a production benchmark** |
| **F1 Score** | **Not treated as a production benchmark** | **Not treated as a production benchmark** |
| **False Acceptance Rate (FAR)** | **Not treated as a production benchmark** | **Not treated as a production benchmark** |
| **False Rejection Rate (FRR)** | **Not treated as a production benchmark** | **Not treated as a production benchmark** |

### Evaluation Diagnostics & Failure Case Analysis
1. **Difficult Image Failure Case**:
   - Sample `V1.jpeg` was intentionally introduced as a challenging, blurred image of registered individual "Varshini".
   - At threshold `0.50`, its similarity was `0.3597`, resulting in a False Rejection (FRR = 33.3%).
   - The threshold sweep revealed that setting the threshold to `0.30` correctly recovers this difficult sample without generating any False Acceptances (FAR remained 0.0%).
2. **Initial Validation Disclaimer**:
   - This evaluation represents an initial validation on a small local dataset, **not** a statistically representative production benchmark.
   - **Reference Leakage Notice**: One test sample (`varshini2.jpg`) was the exact reference image used for registration (similarity `1.0`), serving as a sanity check rather than an out-of-sample test.

---

## 10. Failure Modes & Edge Cases

| Scenario | System Behavior | Rationale |
|---|---|---|
| **Heavy Blur / Low Resolution** | Low similarity or no detection | YuNet/ArcFace cannot reliably extract discriminative features from degraded frames. |
| **Extreme Poses / Profiles** | May fail detection or drop similarity | Frontal/semi-frontal faces are needed for accurate 2D planar embedding generation. |
| **Severe Occlusion (Masks, Glasses)** | Reduced similarity score | Obscured key facial structures reduce cosine similarity below threshold. |
| **Multiple Faces During Registration** | Rejected (`MULTIPLE_FACES`, 400) | Prevents associating an identity with an ambiguous reference crop. |
| **Zero Faces Detected** | Rejected (`NO_FACE`, 400) | Ensures non-face or blank uploads are not saved. |
| **Unknown Intruders** | Classified as `unknown` | Cosine similarity remains well below threshold (`< 0.20` in tests). |

---

## 11. System Limitations

- **Dataset Size**: The current evaluation set is compact and serves for proof-of-concept verification.
- **Single Reference Enrollment**: Currently, one reference crop is stored per identity. Variation in lighting or aging is not yet modeled via multiple centroids.
- **Dataset-Dependent Threshold**: The optimal threshold ($0.30$ vs. default $0.50$) depends on camera quality, resolution, and domain conditions.
- **SQLite / Local Filesystem**: The current deployment uses SQLite and local uploaded images. On Render, the filesystem is ephemeral, so records and uploads can reset after restart or redeploy; managed database and object storage are future production improvements.
- **Cold Starts & Ephemeral Storage on Render Free Tier**:
  - The free-tier container spins down after inactivity; initial requests may take 30–60 seconds.
  - The local container filesystem is ephemeral; SQLite databases reset upon server restart or redeploy.
- **Initial Model Download**: First-time initialization downloads the YuNet ONNX detector and the 249 MB ArcFace ONNX model to the writable runtime cache when they are not bundled.
- **Webcam Scope**: Browser webcam captures client-side still frames sent to `/api/recognize`, whereas local webcam streaming uses `cv2.VideoCapture` via `scripts/recognize_webcam.py`.
- **Free-Tier Deployment**: The hosted deployment is intended for technical demonstration purposes.

---

## 12. Future Improvements

- [ ] **Multi-Reference Enrollment**: Average multiple embeddings per person across angles and expressions.
- [ ] **Automated Threshold Calibration**: Calibration over standard public benchmarks (LFW, CFP-FP).
- [ ] **ROC / DET Analysis**: Continuous trade-off curves for security-critical vs. convenience-oriented applications.
- [ ] **Anti-Spoofing / Liveness Detection**: Texture analysis or blink/motion checks to prevent photo/screen replay attacks.
- [ ] **Cloud Storage & Database**: Transition to AWS S3 / Cloud Storage and PostgreSQL with pgvector.
- [ ] **Startup Pre-warming**: Asynchronously load TensorFlow/DeepFace weights on application startup.
- [ ] **Authentication & Access Control**: JWT-based API authentication and role-based access.
- [ ] **Monitoring & Observability**: Structured request logging, latency metrics, and error tracking.

---

## 13. Technology Stack

- **Backend**: Python 3.11, FastAPI, Uvicorn, Gunicorn, Starlette, Pydantic
- **Computer Vision & ML**: OpenCV (`opencv-python`), DeepFace, ArcFace, TensorFlow, NumPy, Pillow
- **Database**: SQLite3
- **Frontend**: React 18, Vite, Lucide React, Modern Vanilla CSS
- **Testing & Quality**: Pytest, Compileall

---

## 14. Project Structure

```text
face-recognition-project/
├── api/
│   ├── main.py                  # FastAPI application entry point & CORS configuration
│   ├── schemas.py               # Pydantic request/response data schemas
│   └── routes/
│       ├── recognition.py       # POST /api/recognize endpoint
│       ├── people.py            # GET, POST, DELETE /api/people endpoints
│       └── evaluation.py        # GET /api/evaluation endpoint
├── app/
│   ├── config.py                # Central application and path configuration
│   ├── detection/
│   │   └── face_detector.py     # OpenCV YuNet face detector wrapper
│   ├── embeddings/
│   │   └── embedding_generator.py # DeepFace ArcFace embedding generator
│   ├── recognition/
│   │   ├── matcher.py           # Cosine similarity vector matcher
│   │   └── threshold.py         # Similarity threshold decision logic
│   ├── storage/
│   │   ├── database.py          # SQLite database connection & schema
│   │   └── face_repository.py   # Identity CRUD repository
│   ├── workflows/
│   │   ├── image_recognition.py # Still image recognition workflow
│   │   ├── registration.py      # Face registration workflow
│   │   └── webcam_recognition.py# Real-time webcam frame processing
│   └── evaluation/
│       ├── dataset_preparation.py # Dataset partition and label organizer
│       ├── evaluator.py         # Metrics calculation (Accuracy, F1, FAR, FRR)
│       └── image_inventory.py   # Safe directory image indexer
├── frontend/                    # React + Vite web dashboard
│   ├── src/                     # UI components, services, and styling
│   └── package.json
├── scripts/                     # Standalone CLI tools
│   ├── register_person.py       # CLI registration
│   ├── recognize_image.py       # CLI image recognition
│   ├── recognize_webcam.py      # CLI webcam recognition
│   ├── prepare_evaluation_dataset.py # CLI dataset preparation
│   ├── inventory_images.py      # CLI image inventory
│   └── evaluate_system.py       # CLI evaluation runner
├── tests/                       # Pytest test suite (77 tests)
│   ├── api/                     # API integration tests
│   └── test_*.py                # Component unit tests
├── requirements.txt             # Pinned Python dependencies (UTF-8)
├── .python-version              # Python version lock (3.11.9)
└── README.md                    # Project documentation
```

---

## 15. Local Setup & Running

### Prerequisites
- Python 3.11.x installed
- Node.js 18+ and npm installed
- Git

### 1. Clone & Setup Python Environment (Windows)
```powershell
git clone <repository-url>
cd face-recognition-project

# Create virtual environment with Python 3.11
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Backend Server
```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```
- API will be accessible at: `http://127.0.0.1:8000`
- Interactive API Docs (Swagger): `http://127.0.0.1:8000/docs`

### 3. Run Frontend Dashboard
Open a new terminal:
```powershell
cd frontend
npm install
npm run dev
```
- Frontend will be accessible at: `http://localhost:5173`

---

## 16. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check returning service status (`{"status": "ok"}`). |
| `POST` | `/api/recognize` | Upload an image file (`multipart/form-data`) to detect and recognize faces. |
| `GET` | `/api/people` | Retrieve roster of all registered people (omits raw embeddings). |
| `POST` | `/api/people/register` | Register a new identity with `name` (form field) and `file` (image). |
| `DELETE` | `/api/people/{person_id}` | Delete a registered identity and associated embedding from SQLite. |
| `GET` | `/api/evaluation` | Fetch the cached offline evaluation metrics and diagnostics. |

---

## 17. Interactive API Documentation

FastAPI provides automated documentation out of the box:
- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`
- **OpenAPI JSON Schema**: `http://127.0.0.1:8000/openapi.json`

---

## 18. Production Deployment

### Frontend → Vercel
- Root Directory: `frontend`
- Build Command: `npm run build`
- Output Directory: `dist`
- Environment Variable:
  - `VITE_API_BASE_URL`: Public URL of the deployed Render backend (e.g. `https://visionid-api.onrender.com`).

### Backend → Render
- Environment: Python (uses `.python-version` 3.11.9)
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- Environment Variable:
  - `VISIONID_CORS_ORIGINS`: Comma-separated allowed frontend origins (e.g. `https://visionid-murex.vercel.app`).

---

## 19. Hugging Face ZeroGPU Demo Adapter

The `hf_space/` directory contains a Gradio adapter for a Hugging Face Space. It calls the existing VisionID workflows and preserves YuNet, DeepFace ArcFace, 512-dimensional embeddings, SQLite storage, cosine matching, and the authoritative 0.50 threshold. It does not introduce a second recognition implementation or mock results.

The Space is a sample/demo deployment. Free Space storage is not production-persistent, so SQLite registrations may be lost after rebuilds or lifecycle events. TensorFlow/DeepFace/ArcFace compatibility with the Space's ZeroGPU runtime must be verified at runtime; a successful build alone is not acceptance. If model initialization or inference fails, use the documented blocker rather than replacing the required pipeline.

## 20. Environment Variables

| Variable | Scope | Description | Default |
|---|---|---|---|
| `VITE_API_BASE_URL` | Frontend | Target backend API base URL | `http://127.0.0.1:8000` |
| `VISIONID_CORS_ORIGINS` | Backend | Allowed CORS origins for cross-origin browser requests | `http://localhost:5173,http://127.0.0.1:5173` |
| `FACE_RECOGNITION_DB_PATH` | Backend | Custom location for the SQLite database file | `data/face_recognition.db` |
| `REGISTERED_FACES_DIR` | Backend | Directory for stored registration reference images | `data/registered_faces` |

*(No private secrets or API keys are required to operate the core recognition engine.)*

---

## 20. Automated Testing & Verification

Run the full automated test suite using the project virtual environment:
```powershell
.\venv\Scripts\python.exe -m pytest -q
```
**Current Status**:
```text
77 passed, 1 warning in 2.51s
```

Verify clean bytecode compilation:
```powershell
.\venv\Scripts\python.exe -m compileall app api
```

---

## 21. Security & Privacy

- **Data Privacy**: Raw face embeddings and registration reference images are stored locally in `data/` and strictly excluded from version control via `.gitignore`.
- **Error Sanitization**: Server-side exceptions are logged with full stack traces in server logs, while API responses return clean, generic error codes (`PROCESSING_ERROR`, `DATABASE_ERROR`) without leaking internal system traces.
- **Upload Lifecycle**: All temporary files generated during multipart uploads are guaranteed to be unlinked in `finally` blocks.
- **Credential Safety**: No API keys, passwords, tokens, or environment credentials are tracked in the repository.

---

## 22. Submission Checklist

- [x] **Face Detection**: OpenCV YuNet ONNX detector integrated and passing tests.
- [x] **Face Embeddings**: DeepFace ArcFace 512-D normalized vector representation.
- [x] **Similarity Matching**: Cosine similarity implementation with scalar output.
- [x] **Unknown Face Rejection**: Enforced similarity threshold ($0.50$ default).
- [x] **Evaluation Suite**: Offline evaluation tool with precision, recall, F1, FAR, FRR, and threshold sweeps.
- [x] **REST API**: FastAPI with health, recognition, people management, and evaluation routes.
- [x] **Web Frontend**: React + Vite dashboard with live pipeline feedback and person roster.
- [x] **Test Coverage**: 77 unit, integration, and API tests passing with 0 failures.
- [x] **Documentation**: Full architectural rationale, failure modes, setup, and deployment guide.
- [x] **Production Ready**: Verified build, UTF-8 requirements, CORS configuration, and clean git state.
