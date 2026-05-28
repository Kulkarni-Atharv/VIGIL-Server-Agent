"""
Camera Stream Manager — opens and reads real RTSP streams on this Jetson.

Each assigned camera gets its own background thread that continuously reads
frames via OpenCV VideoCapture. This drives real CPU/GPU load so the central
server's load balancer sees genuine metric changes.

If cv2 is not installed the manager falls back to stub mode (no decoding),
so the code runs on a dev PC without OpenCV.

Stage 3: swap the cv2 loop for a DeepStream/GStreamer inference pipeline.
"""

import logging
import threading
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.warning("cv2 not found — streams will run in stub mode (no frame decoding)")


class _StreamWorker:
    """Reads frames from one RTSP stream in a daemon thread."""

    def __init__(self, camera_id: str, rtsp_url: str):
        self.camera_id = camera_id
        self.rtsp_url  = rtsp_url
        self._stop     = threading.Event()
        self._lock     = threading.Lock()
        self._fps      = 0.0
        self._errors   = 0
        self._thread   = threading.Thread(
            target=self._run, name=f"stream-{camera_id}", daemon=True
        )
        self._thread.start()

    def _run(self):
        if not _CV2_AVAILABLE:
            self._run_stub()
            return

        logger.info(f"[{self.camera_id}] Opening RTSP: {self.rtsp_url}")
        cap = cv2.VideoCapture(self.rtsp_url)

        if not cap.isOpened():
            logger.error(f"[{self.camera_id}] Could not open stream — check RTSP URL")
            with self._lock:
                self._errors += 1
            return

        frame_count = 0
        t_start = time.monotonic()

        while not self._stop.is_set():
            ret, _frame = cap.read()

            if not ret:
                with self._lock:
                    self._errors += 1
                logger.warning(f"[{self.camera_id}] Frame read failed — reconnecting in 2s")
                cap.release()
                time.sleep(2)
                if self._stop.is_set():
                    break
                cap = cv2.VideoCapture(self.rtsp_url)
                frame_count = 0
                t_start = time.monotonic()
                continue

            frame_count += 1
            elapsed = time.monotonic() - t_start
            # Recalculate FPS every second
            if elapsed >= 1.0:
                with self._lock:
                    self._fps = round(frame_count / elapsed, 1)
                frame_count = 0
                t_start = time.monotonic()

        cap.release()
        logger.info(f"[{self.camera_id}] Stream stopped")

    def _run_stub(self):
        """Stub mode — no real decoding, simulates 25 fps so metrics still work."""
        logger.info(f"[{self.camera_id}] Stub stream started (cv2 not available)")
        with self._lock:
            self._fps = 25.0
        while not self._stop.is_set():
            time.sleep(0.04)

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

    def get_average_fps(self) -> Optional[float]:
        with self._lock:
            workers = list(self._workers.values())
        if not workers:
            return None
        return round(sum(w.fps for w in workers) / len(workers), 1)

    def get_total_errors(self) -> int:
        with self._lock:
            workers = list(self._workers.values())
        return sum(w.errors for w in workers)

    def get_all(self) -> dict:
        with self._lock:
            return {
                cid: {"fps": w.fps, "errors": w.errors}
                for cid, w in self._workers.items()
            }
