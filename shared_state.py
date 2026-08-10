"""
Shared state module — holds the single stream-manager instance shared
between the command receiver and the telemetry agent. Both import
`stream_mgr` from here so they operate on the same object.

Two backends, selected via STREAM_BACKEND (default: opencv):
  opencv    → camera_stream_manager.py    (unchanged, default)
  deepstream→ deepstream_stream_manager.py (see that file's docstring)

Both expose the exact same interface (start_stream/stop_stream/
get_camera_health/get_active_camera_ids/...), so command_receiver.py and
jetson_main.py work unmodified regardless of which backend is active — and
the central server's contract is identical either way.
"""

import os

STREAM_BACKEND = os.getenv("STREAM_BACKEND", "opencv").strip().lower()

if STREAM_BACKEND == "deepstream":
    from deepstream_stream_manager import DeepStreamStreamManager
    stream_mgr = DeepStreamStreamManager()
elif STREAM_BACKEND == "opencv":
    from camera_stream_manager import CameraStreamManager
    stream_mgr = CameraStreamManager()
else:
    raise SystemExit(
        f"ERROR: STREAM_BACKEND='{STREAM_BACKEND}' is not valid — use 'opencv' or 'deepstream'"
    )
