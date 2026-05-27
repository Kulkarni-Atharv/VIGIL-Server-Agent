"""
Camera Stream Manager — tracks active RTSP streams on this Jetson.

Stage 2: This is a functional stub. It manages an in-memory registry of
streams and provides real metrics to the telemetry agent.

Stage 3: Replace start_stream() / stop_stream() with real
DeepStream / GStreamer pipeline calls.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class CameraStreamManager:
    def __init__(self):
        # camera_id → {"rtsp_url": str, "fps": float, "errors": int}
        self._streams: Dict[str, dict] = {}

    def start_stream(self, camera_id: str, rtsp_url: str) -> bool:
        """
        Register a camera stream as active.
        Stage 3: launch the actual DeepStream pipeline here.
        Returns True on success.
        """
        if camera_id in self._streams:
            logger.warning(f"Camera {camera_id} already active — replacing entry")
        self._streams[camera_id] = {
            "rtsp_url": rtsp_url,
            "fps":      25.0,   # Stage 3: read real FPS from DeepStream pipeline
            "errors":   0,
        }
        logger.info(f"Stream STARTED: {camera_id} → {rtsp_url}")
        return True

    def stop_stream(self, camera_id: str) -> None:
        """
        Deregister a camera stream.
        Stage 3: tear down the DeepStream pipeline here.
        """
        if camera_id in self._streams:
            del self._streams[camera_id]
            logger.info(f"Stream STOPPED: {camera_id}")
        else:
            logger.warning(f"Stop requested for unknown camera: {camera_id}")

    def get_camera_count(self) -> int:
        return len(self._streams)

    def get_average_fps(self) -> Optional[float]:
        if not self._streams:
            return None
        return round(
            sum(s["fps"] for s in self._streams.values()) / len(self._streams), 1
        )

    def get_total_errors(self) -> int:
        return sum(s["errors"] for s in self._streams.values())

    def get_all(self) -> dict:
        return dict(self._streams)
