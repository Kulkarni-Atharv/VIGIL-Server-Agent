# VIGIL Jetson Agent

The edge component of the VIGIL factory-safety system. Runs on every NVIDIA Jetson device on the factory floor and does two things simultaneously:

1. **Sends telemetry heartbeats** — CPU, RAM, disk, GPU usage, temperature, active camera count, FPS, and RTSP error count — to the Central Server every 2 seconds so the load balancer has live data.
2. **Listens for commands** — accepts `ASSIGN_CAMERA` and `REMOVE_CAMERA` instructions from the Central Server and manages the RTSP stream workers accordingly.

---

## Architecture

```
jetson_main.py
├── Thread 1 — Command Receiver (FastAPI on port 8001)
│   └── command_receiver.py  →  shared_state.stream_mgr
│
└── Thread 2 — Heartbeat Loop (main thread)
    └── agent.py metrics  +  shared_state.stream_mgr live stats
                               │
                               ▼
                    camera_stream_manager.py
                    (one background thread per RTSP stream)
```

Both threads share a single `CameraStreamManager` instance via `shared_state.py` so the heartbeat always reports live stream metrics.

---

## File Overview

| File | Role |
|------|------|
| `jetson_main.py` | Entry point — starts command receiver thread + heartbeat loop |
| `agent.py` | Standalone Stage 1 heartbeat agent (metrics only, no camera management) |
| `command_receiver.py` | FastAPI server on port 8001 — handles `ASSIGN_CAMERA` / `REMOVE_CAMERA` |
| `camera_stream_manager.py` | Thread-safe manager for RTSP stream workers (OpenCV) |
| `shared_state.py` | Singleton `CameraStreamManager` shared across both threads |

---

## Requirements

### Hardware
- NVIDIA Jetson (Nano, NX, AGX, Orin) with JetPack 4.6 or later
- Network access to the PC running the Central Server

### Software
- Python 3.8+
- OpenCV — **already included in JetPack**, do not `pip install opencv-python` on a real Jetson (it will conflict with the hardware-accelerated system build)
- On a dev PC without JetPack, uncomment `opencv-python` in `requirements.txt`

---

## Setup

**1. Clone and install dependencies**

```bash
git clone <repo-url>
cd jetson_agent
pip install -r requirements.txt
```

> On a real Jetson, OpenCV is provided by JetPack. Leave it commented in `requirements.txt`.

**2. Create your `.env` file**

```bash
cp .env.example .env
nano .env
```

Fill in the values:

```env
# IP of the PC running the Central Server
SERVER_URL=http://192.168.1.100:8000

# Unique identifier for this Jetson (use a different name on each device)
JETSON_ID=jetson-01

# Must match API_KEY in central_server/.env
API_KEY=your-shared-secret-key

# Seconds between heartbeats (default: 2)
HEARTBEAT_INTERVAL=2

# Set true on PC/VM to simulate GPU & temperature instead of calling tegrastats
MOCK_JETSON_METRICS=false
```

> `JETSON_ID` must be unique across every Jetson on your network — the Central Server uses it as the primary key.

---

## Running

```bash
python jetson_main.py
```

Expected startup output:

```
2025-01-01 12:00:00 [INFO] VIGIL Jetson starting | ID:jetson-01
2025-01-01 12:00:00 [INFO] Command receiver listening on port 8001
2025-01-01 12:00:00 [INFO] Heartbeat loop starting | ID:jetson-01 | Server:http://192.168.1.100:8000 | Interval:2s
2025-01-01 12:00:02 [INFO] Heartbeat OK | CPU:12.3% RAM:45.1% GPU:18.0 Temp:41.5°C Cams:0 → ONLINE (score 18.4)
```

---

## How It Works

### Telemetry Collection

Every `HEARTBEAT_INTERVAL` seconds the agent collects:

| Metric | Source |
|--------|--------|
| `cpu_percent` | `psutil.cpu_percent()` — 1-second blocking sample |
| `ram_percent` | `psutil.virtual_memory().percent` |
| `disk_percent` | `psutil.disk_usage("/").percent` |
| `gpu_percent` | `tegrastats` — parsed from `GR3D_FREQ` field |
| `temperature` | `tegrastats` — parsed from `gpu@`, `tj@`, or `cpu@` sensor |
| `assigned_camera_count` | `CameraStreamManager.get_camera_count()` |
| `detection_fps` | `CameraStreamManager.get_average_fps()` — average across all active streams |
| `rtsp_error_count` | `CameraStreamManager.get_total_errors()` — cumulative frame-read failures |

The payload is POSTed to `POST /api/v1/heartbeat` on the Central Server with the shared `API_KEY` in the `x-api-key` header. The server returns the computed `load_status` (`ONLINE` / `DEGRADED`) and `load_score` which are logged on each heartbeat.

### GPU & Temperature via `tegrastats`

On Jetson hardware, `tegrastats` is the only way to read GPU utilisation. The agent spawns it with `--interval 500`, reads one output line, then terminates the process. It tries three temperature sensor labels in order: `gpu@` → `tj@` (junction/hottest point) → `cpu@`.

If `tegrastats` is not found (dev PC), both values are reported as `null`.

### Camera Stream Management

When the Central Server sends `ASSIGN_CAMERA`, the agent:

1. Creates a `_StreamWorker` thread for that camera
2. Opens the RTSP URL with `cv2.VideoCapture`
3. Reads frames in a loop, recalculating FPS every second
4. On a failed read, waits 2 seconds and reconnects automatically
5. Increments `_errors` on each failed read — this feeds back into the load score

When the Central Server sends `REMOVE_CAMERA`, the worker thread is signalled to stop and the `VideoCapture` is released cleanly.

**Stub mode** — if `cv2` is not installed (dev PC without JetPack), the manager runs in stub mode: no frame decoding, reports a constant 25 FPS so the rest of the pipeline still works.

### Command Receiver (Port 8001)

A FastAPI server that accepts commands from the Central Server:

| Command | Required fields | Action |
|---------|----------------|--------|
| `ASSIGN_CAMERA` | `camera_id`, `rtsp_url` | Opens RTSP stream |
| `REMOVE_CAMERA` | `camera_id` | Stops and removes stream |
| `GET_STATUS` | — | Returns active stream count and per-stream FPS/errors |

All requests must include the `x-api-key` header with the shared secret, otherwise the endpoint returns `401`.

---

## Development / Testing on PC

Set `MOCK_JETSON_METRICS=true` in `.env`. The agent will generate random GPU (15–75%) and temperature (36–65°C) values instead of calling `tegrastats`. All other behaviour is identical.

To also test RTSP streams on PC, uncomment `opencv-python` in `requirements.txt` and install it, then provide valid RTSP URLs when registering cameras on the Central Server.

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_URL` | — | Full URL of the Central Server, e.g. `http://192.168.1.100:8000` |
| `JETSON_ID` | — | Unique ID for this device, e.g. `jetson-01` |
| `API_KEY` | — | Shared secret — must match `API_KEY` in `central_server/.env` |
| `HEARTBEAT_INTERVAL` | `2` | Seconds between heartbeats |
| `CMD_RECEIVER_PORT` | `8001` | Port the command receiver listens on |
| `MOCK_JETSON_METRICS` | `false` | Set `true` to simulate GPU/temp on non-Jetson hardware |

---

## Multi-Jetson Deployment

Run `jetson_main.py` on each Jetson. The only difference per device is `JETSON_ID` — give every device a unique name (`jetson-01`, `jetson-02`, …). They all point at the same `SERVER_URL` and use the same `API_KEY`.

```
Factory Floor
├── Jetson-01  →  jetson_main.py  →  Central Server (192.168.1.100:8000)
├── Jetson-02  →  jetson_main.py  →  Central Server
└── Jetson-03  →  jetson_main.py  →  Central Server
```

The Central Server's load balancer distributes cameras across all connected Jetsons automatically.

---

## Stage 3 — DeepStream / Inference Pipeline

`camera_stream_manager.py` currently uses a plain OpenCV `VideoCapture` loop. The Stage 3 upgrade replaces `_StreamWorker._run()` with a GStreamer / NVIDIA DeepStream pipeline for hardware-accelerated inference (object detection, pose estimation, etc.) — all other components remain unchanged.
