# VisionID Frontend

The frontend foundation is a React/Vite product shell for the existing Python face-recognition engine. It intentionally has no backend connection yet and does not perform recognition in the browser.

## Commands

```powershell
npm install
npm run dev
npm run build
```

The next integration phase can connect `src/services/recognitionService.js` to a real API without moving recognition logic into JavaScript.
