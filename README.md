# Stormwater Monitoring System

A real-time stormwater pollution monitoring prototype. The system monitors water quality at stormwater drain outlets during first-flush rain events, detecting elevated turbidity and conductivity and triggering a physical diverter response when pollution thresholds are exceeded.

## Architecture

```
backend/    FastAPI server — sensor reading, SQLite storage, WebSocket broadcasting
frontend/   Vue 3 + Vuetify dashboard — live readings, history chart, event log
```

## Stack

- **Backend:** Python, FastAPI, SQLite, Uvicorn
- **Frontend:** Vue 3, Vuetify, ApexCharts, Axios
- **Hardware (planned):** Raspberry Pi 4, CSI camera module, turbidity sensor, conductivity sensor, GPIO LED

## Getting started

### Prerequisites

- Python 3.10+
- Node.js 18+

### Backend

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / Pi
pip install -r backend/requirements.txt
make backend
```

### Frontend

```bash
cd frontend && npm install
make frontend
```

Open `http://localhost:5173` in your browser.

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | Current sensor snapshot |
| GET | `/api/readings` | Historical readings from SQLite |
| GET | `/api/events` | Threshold breach event log |
| GET | `/api/vision/health` | OpenCV service status |
| POST | `/api/vision/analyze` | Analyse a JPEG, PNG, or WEBP image (`image` multipart field) |
| WS | `/ws/live` | Live sensor stream at 2s intervals |

The vision endpoint uses lightweight deterministic OpenCV threshold and contour
screening for Raspberry Pi deployment. It is not a trained ML model and cannot
determine whether water is safe or identify dissolved or microbial pollution.

## Production

Build the frontend and serve static files via FastAPI:

```bash
cd frontend && npm run build
# copy dist/ to backend/static/
make backend
```

Then open `http://<device-ip>:8000` from any device on the same network.
