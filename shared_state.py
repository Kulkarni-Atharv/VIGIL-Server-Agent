"""
Shared state module — holds the single CameraStreamManager instance
that is shared between the command receiver and the telemetry agent.
Both import `stream_mgr` from here so they operate on the same object.
"""

from camera_stream_manager import CameraStreamManager

stream_mgr = CameraStreamManager()
