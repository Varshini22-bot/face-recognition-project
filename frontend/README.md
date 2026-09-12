# VisionID Frontend

The frontend is a React/Vite dashboard for the existing Python face-recognition engine. Recognition stays on the FastAPI backend; the browser uploads images and renders API responses without running ML locally.

## API routing

- Local development uses `VITE_API_BASE_URL` when set, otherwise `http://127.0.0.1:8000`.
- Production uses the Cloudflare Quick Tunnel configured in `VITE_API_BASE_URL`.
- The service layer owns route normalization, multipart uploads, and consistent error messages.

## Commands

```powershell
npm install
npm run dev
npm run build
```

The next integration phase can connect `src/services/recognitionService.js` to a real API without moving recognition logic into JavaScript.
