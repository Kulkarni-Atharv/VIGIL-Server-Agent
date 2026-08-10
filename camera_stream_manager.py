"""
Camera Stream Manager — opens and reads real RTSP streams on this Jetson.

Each assigned camera gets its own background thread that continuously reads
frames via OpenCV VideoCapture. This drives real CPU/GPU load so the central
server's load balancer sees genuine metric changes.

If cv2 is not installed the manager falls back to stub mode (no decoding),
so the code runs on a dev PC without OpenCV.

── Camera health, decoupled from AI inference rate ───────────────────────────
Health is based on *frame staleness* — how long it's been since cap.read()
last succeeded — NOT on detection FPS. A pipeline doing 0.2 fps AI inference
still reads raw frames continuously; health tracks that raw read loop, so a
deliberately slow/throttled AI stage never gets mistaken for a dead camera.

Three states per camera:
  STARTING — assigned, hasn't produced a first frame yet
  UP       — a frame was read within CAMERA_STALE_AFTER_SECONDS
  DOWN     — no frame for that long, or too many consecutive read failures

Stage 3: swap the cv2 loop for a DeepStream/GStreamer inference pipeline.
"""

import logging
import os
import threading
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.warning("cv2 not found — streams will run in stub mode (no frame decoding)")

# ── Camera health thresholds (independent of AI inference rate) ──────────────
# How long a stream can go without a successful frame read before it's DOWN.
# Tune this to your camera's actual frame-read cadence, not your AI FPS.
CAMERA_STALE_AFTER_SECONDS = float(os.getenv("CAMERA_STALE_AFTER_SECONDS", "30"))
# Consecutive failed reads before immediately flagging DOWN (doesn't wait
# for the staleness window if the connection is visibly broken).
CAMERA_FAIL_THRESHOLD      = int(os.getenv("CAMERA_FAIL_THRESHOLD", "5"))
# Backoff between reconnect attempts after a read failure.
CAMERA_RECONNECT_DELAY     = float(os.getenv("CAMERA_RECONNECT_DELAY_SECONDS", "2"))


class _StreamWorker:
    """Reads frames from one RTSP stream in a daemon thread."""

    def __init__(self, camera_id: str, rtsp_url: str):
        self.camera_id = camera_id
        self.rtsp_url  = rtsp_url
        self._stop     = threading.Event()
        self._lock     = threading.Lock()
        self._fps      = 0.0
        self._errors   = 0                             # lifetime count — informational only
        self._consecutive_failures = 0                  # reset on every successful read
        self._last_frame_at: Optional[float] = None     # monotonic seconds
        self._thread   = threading.Thread(
            target=self._run, name=f"stream-{camera_id}", daemon=True
        )
        self._thread.start()

    def _open(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            with self._lock:
                self._errors += 1
                self._consecutive_failures += 1
        return cap

    def _run(self):
        if not _CV2_AVAILABLE:
            self._run_stub()
            return

        logger.info(f"[{self.camera_id}] Opening RTSP: {self.rtsp_url}")
        cap = self._open()

        frame_count = 0
        t_start = time.monotonic()

        while not self._stop.is_set():
            if not cap.isOpened():
                logger.warning(
                    f"[{self.camera_id}] Stream not open — retrying in "
                    f"{CAMERA_RECONNECT_DELAY}s (consecutive failures: "
                    f"{self._consecutive_failures})"
                )
                time.sleep(CAMERA_RECONNECT_DELAY)
                if self._stop.is_set():
                    break
                cap = self._open()
                continue

            ret, _frame = cap.read()

            if not ret:
                with self._lock:
                    self._errors += 1
                    self._consecutive_failures += 1
                logger.warning(
                    f"[{self.camera_id}] Frame read failed — reconnecting in "
                    f"{CAMERA_RECONNECT_DELAY}s (consecutive failures: "
                    f"{self._consecutive_failures})"
                )
                cap.release()
                time.sleep(CAMERA_RECONNECT_DELAY)
                if self._stop.is_set():
                    break
                cap = self._open()
                frame_count = 0
                t_start = time.monotonic()
                continue

            # Successful read — this is the health signal, independent of
            # whatever the AI stage does with the frame afterwards.
            with self._lock:
                self._last_frame_at = time.monotonic()
                self._consecutive_failures = 0

            frame_count += 1
            elapsed = time.monotonic() - t_start
            # Recalculate FPS every second — informational metric only,
            # never used for health/DEGRADED decisions.
            if elapsed >= 1.0:
                with self._lock:
                    self._fps = round(frame_count / elapsed, 1)
                frame_count = 0
                t_start = time.monotonic()

        cap.release()
        logger.info(f"[{self.camera_id}] Stream stopped")

    def _run_stub(self):
        """Stub mode — no real decoding, simulates a healthy feed so metrics still work."""
        logger.info(f"[{self.camera_id}] Stub stream started (cv2 not available)")
        with self._lock:
            self._fps = 25.0
            self._last_frame_at = time.monotonic()
        while not self._stop.is_set():
            time.sleep(0.5)
            with self._lock:
                self._last_frame_at = time.monotonic()

    def stop(self):
        self._stop.set()

    @property
    def fps(self) -> float:
        with self._lock:
            return self._fps

    @property
    def errors(self) -> int:
        with self._lock:
            return self._errors

    def health(self, now: Optional[float] = None) -> str:
        """Return STARTING / UP / DOWN based on frame staleness — never on FPS."""
        now = now if now is not None else time.monotonic()
        with self._lock:
            if self._consecutive_failures >= CAMERA_FAIL_THRESHOLD:
                return "DOWN"
            if self._last_frame_at is None:
                return "STARTING"
            if now - self._last_frame_at > CAMERA_STALE_AFTER_SECONDS:
                return "DOWN"
            return "UP"

    def last_frame_age(self, now: Optional[float] = None) -> Optional[float]:
        now = now if now is not None else time.monotonic()
        with self._lock:
            if self._last_frame_at is None:
                return None
            return round(now - self._last_frame_at, 1)


class CameraStreamManager:

    def __init__(self):
        self._workers: Dict[str, _StreamWorker] = {}
        self._lock = threading.Lock()

    def start_stream(self, camera_id: str, rtsp_url: str) -> bool:
        """
        Open an RTSP stream for camera_id in a background thread.
        If the camera is already active, stops the old worker first.
        Returns True immediately (stream opens asynchronously).
        """
        with self._lock:
            if camera_id in self._workers:
                logger.warning(f"Camera {camera_id} already active — replacing worker")
                self._workers[camera_id].stop()
            self._workers[camera_id] = _StreamWorker(camera_id, rtsp_url)
        logger.info(f"Stream STARTED: {camera_id} → {rtsp_url}")
        return True

    def stop_stream(self, camera_id: str) -> None:
        """Stop and remove the stream worker for camera_id."""
        with self._lock:
            worker = self._workers.pop(camera_id, None)
        if worker:
            worker.stop()
            logger.info(f"Stream STOPPED: {camera_id}")
        else:
            logger.warning(f"Stop requested for unknown camera: {camera_id}")

    def get_camera_count(self) -> int:
        with self._lock:
            return len(self._workers)

    def get_active_camera_ids(self) -> List[str]:
        """All camera_ids this Jetson currently believes it's streaming.
        Used by the central server to reconcile stale connections after a
        network partition (see main.py's heartbeat reconciliation step)."""
        with self._lock:
            return list(self._workers.keys())

    def get_average_fps(self) -> Optional[float]:
        """Informational only — NOT used for health/DEGRADED decisions."""
        with self._lock:
            workers = list(self._workers.values())
        if not workers:
            return None
        return round(sum(w.fps for w in workers) / len(workers), 1)

    def get_total_errors(self) -> int:
        """Informational only — NOT used for health/DEGRADED decisions."""
        with self._lock:
            workers = list(self._workers.values())
        return sum(w.errors for w in workers)

    def get_camera_health(self) -> Dict[str, str]:
        """{camera_id: 'STARTING'|'UP'|'DOWN'} — the actual health signal,
        based on frame staleness, sent to the central server every heartbeat."""
        now = time.monotonic()
        with self._lock:
            workers = dict(self._workers)
        return {cid: w.health(now) for cid, w in workers.items()}

    def get_all(self) -> dict:
        now = time.monotonic()
        with self._lock:
            workers = dict(self._workers)
        return {
            cid: {
                "fps": w.fps,
                "errors": w.errors,
                "health": w.health(now),
                "last_frame_age_seconds": w.last_frame_age(now),
            }
            for cid, w in workers.items()
        }
