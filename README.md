# Stormwater Monitoring System

## Trained waste detection (camera evaluation branch)

The existing browser camera/upload page is retained for evaluation. The default
vision engine is now a local YOLOv8 ONNX waste detector from
[ditoow/trashscan8s](https://huggingface.co/ditoow/trashscan8s), with
[trashscan8n](https://huggingface.co/ditoow/trashscan8n) available for smaller devices.
Both expose paper, plastic, metal, organic and other classes. These are community
models, not validated Melbourne floodwater models. Model cards declare MIT;
review upstream YOLO licensing for your intended use. No images are sent to a cloud API.

After installing backend requirements, run from the repository root:

```sh
python backend/download_waste_model.py --size s
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

The downloader reports the source revision and SHA256. Weights are ignored by
Git and must be downloaded on each device. API errors explicitly report missing
weights/dependencies; there is no silent fallback to colour thresholds.

For a 64-bit Raspberry Pi OS environment, install requirements and download the
nano variant, then start from the repository root:

```sh
python backend/download_waste_model.py --size n
export WASTE_MODEL_PATH="$PWD/backend/weights/trashscan8n.onnx"
export WASTE_THREADS=2
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

ONNX Runtime CPU inference needs no GPU or PyTorch. Actual Pi memory use and
latency still require hardware testing. The browser submits sampled frames, not
30 FPS detection; `inferenceMs` is included in each response. Set
`WASTE_CONFIDENCE=0.40` to tune the detection threshold. `VISION_ENGINE=opencv`
selects the prior baseline for comparison. Restart after configuration changes.

The `score` is an illustrative visible-litter index, not pollution probability.
Organic/other objects are displayed but excluded from this index; turbidity and
chemical quality are not measured by this detector. Detections may be on shore
because this model does not segment water. Keep a separate labelled evaluation
set of clear water, reflections, vegetation and waste before claiming accuracy.

The current frontend remains a browser-camera test harness. CSI/Picamera2 capture
and the team's final main-branch frontend are separate integration work; they
can submit JPEGs to the same POST `/api/vision/analyze` contract.

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
| GET | `/api/vision/health` | Vision engine and model status |
| POST | `/api/vision/analyze` | Analyse a JPEG, PNG, or WEBP image (`image` multipart field) |
| WS | `/ws/live` | Live sensor stream at 2s intervals |

The vision endpoint defaults to the trained waste model described above; the
OpenCV threshold baseline remains optional. Neither determines whether water is
safe or identifies dissolved or microbial pollution.
The Vue Camera view can open the browser camera, analyse a frame every 2.5
seconds, draw returned anomaly boxes, or analyse an uploaded image.

## Production

Build the frontend and serve static files via FastAPI:

```bash
cd frontend && npm run build
# copy dist/ to backend/static/
make backend
```

Then open `http://<device-ip>:8000` from any device on the same network.
