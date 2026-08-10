"""
DeepStream Stream Manager — GStreamer/DeepStream equivalent of
camera_stream_manager.py. Same public interface, same health contract sent
to the central server (camera_health / active_camera_ids) — the central
server does not know or care which of the two is running underneath.

Kept as a SEPARATE module from camera_stream_manager.py on purpose — the
OpenCV path is untouched. Pick one via shared_state.py's STREAM_BACKEND
env var.

── Why this looks different from the OpenCV version ──────────────────────────
OpenCV gives a simple `ret, frame = cap.read()` per camera. DeepStream has
no such per-camera call — many cameras are batched through one shared
GStreamer pipeline (nvstreammux → your inference branch), so "did camera X
just produce a frame" has to be observed a different way:

  1. Bus messages   — ERROR/EOS on a source's decode bin → that source is
                       DOWN immediately (the connection itself broke).
  2. A pad probe on nvstreammux's src pad — for every batched buffer, read
     each frame's source_id via pyds and stamp "last seen" for that
     source. Any source_id absent for too long → DOWN. This mirrors
     last_frame_at in the OpenCV version, but is read off the *decoder/mux*
     stage, deliberately BEFORE any inference throttling — so, exactly as
     with OpenCV, a slow/throttled AI stage (e.g. 0.2 fps) can never look
     like a dead camera.

── What this module owns vs. what it doesn't ─────────────────────────────────
This manager owns ingestion (per-camera source bins → nvstreammux) and
health tracking only — the same scope as camera_stream_manager.py's OpenCV
loop. It deliberately does NOT build your inference branch (nvinfer /
tracker / OSD / sink), since that's specific to your existing DeepStream
app and model config. Pass `on_source_added(streammux, pad, source_id)` to
the constructor to attach your existing inference branch the first time a
source is linked — see the __main__ example at the bottom of this file.

── Connection budget ──────────────────────────────────────────────────────────
Exactly 1 RTSP connection per camera, same as OpenCV — each camera gets its
own decode bin (uridecodebin) with its own network socket. nvstreammux
multiplexes decoded *frames* into a shared batch; it does not share or
reduce the number of RTSP connections.

NOTE: this file requires GStreamer + the DeepStream Python bindings (pyds),
which only exist on a real Jetson/DeepStream install. It cannot be
exercised on a plain dev machine — if gi/pyds aren't importable, this
module falls back to the same kind of stub mode camera_stream_manager.py
uses for cv2, so imports and unit tests elsewhere in the codebase don't
break. Validate on real hardware before relying on it in production.
"""

import logging
import os
import threading
import time
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import GLib, Gst
    import pyds
    _DEEPSTREAM_AVAILABLE = True
    Gst.init(None)
except (ImportError, ValueError):
    _DEEPSTREAM_AVAILABLE = False
    logger.warning(
        "gi/GStreamer/pyds not found — DeepStreamStreamManager will run in "
        "stub mode (no real pipeline). Only usable on a real Jetson with "
        "DeepStream installed."
    )

# ── Camera health thresholds — identical semantics to camera_stream_manager.py.
# Independent of AI inference rate on purpose (see module docstring).
CAMERA_STALE_AFTER_SECONDS = float(os.getenv("CAMERA_STALE_AFTER_SECONDS", "30"))
CAMERA_FAIL_THRESHOLD      = int(os.getenv("CAMERA_FAIL_THRESHOLD", "5"))

# nvstreammux tuning — override per deployment. Keep in sync with whatever
# your existing DeepStream app already uses for these.
MUX_WIDTH       = int(os.getenv("DS_MUX_WIDTH", "1920"))
MUX_HEIGHT      = int(os.getenv("DS_MUX_HEIGHT", "1080"))
MUX_BATCH_SIZE  = int(os.getenv("DS_MUX_BATCH_SIZE", "12"))
MUX_BATCH_TIMEOUT_USEC = int(os.getenv("DS_MUX_BATCH_TIMEOUT_USEC", "40000"))


class _SourceState:
    """Health bookkeeping for one camera's source bin."""

    __slots__ = ("camera_id", "rtsp_url", "bin", "pad_index",
                 "last_seen_at", "consecutive_failures", "removing")

    def __init__(self, camera_id: str, rtsp_url: str, bin_, pad_index: int):
        self.camera_id  = camera_id
        self.rtsp_url   = rtsp_url
        self.bin        = bin_
        self.pad_index  = pad_index          # nvstreammux sink pad / source_id
        self.last_seen_at: Optional[float] = None
        self.consecutive_failures = 0
        self.removing = False                # True while an EOS-based removal is in flight


class DeepStreamStreamManager:
    """
    Same public interface as CameraStreamManager:
      start_stream(camera_id, rtsp_url) -> bool
      stop_stream(camera_id) -> None
      get_camera_count() -> int
      get_active_camera_ids() -> List[str]
      get_average_fps() -> Optional[float]      (informational only)
      get_total_errors() -> int                 (informational only)
      get_camera_health() -> Dict[str, str]      ("STARTING"/"UP"/"DOWN")
      get_all() -> dict
    """

    def __init__(self, on_source_added: Optional[Callable] = None):
        """
        on_source_added(pipeline, streammux, source_id): called the first
        time a source's pad is linked into streammux, so you can attach your
        existing inference branch (nvinfer/tracker/OSD/sink) here. Left
        unset, the pipeline still runs and reports health — it just has no
        inference branch, which is fine for testing ingestion/health alone.
        """
        self._lock = threading.Lock()
        self._sources: Dict[str, _SourceState] = {}
        self._next_pad_index = 0
        self._on_source_added = on_source_added
        self._loop_thread: Optional[threading.Thread] = None
        self._glib_loop = None

        if not _DEEPSTREAM_AVAILABLE:
            self._stub_health: Dict[str, str] = {}
            return

        self.pipeline = Gst.Pipeline.new("vigil-deepstream-pipeline")
        self.streammux = Gst.ElementFactory.make("nvstreammux", "vigil-streammux")
        self.streammux.set_property("width", MUX_WIDTH)
        self.streammux.set_property("height", MUX_HEIGHT)
        self.streammux.set_property("batch-size", MUX_BATCH_SIZE)
        self.streammux.set_property("batched-push-timeout", MUX_BATCH_TIMEOUT_USEC)
        self.streammux.set_property("live-source", 1)
        self.pipeline.add(self.streammux)

        # Health probe — reads NvDsFrameMeta.source_id off every batched
        # buffer, BEFORE any inference. This is the DeepStream analog of
        # OpenCV's last_frame_at: it reflects raw decode/ingest health, not
        # your (possibly throttled) inference throughput.
        mux_src_pad = self.streammux.get_static_pad("src")
        mux_src_pad.add_probe(Gst.PadProbeType.BUFFER, self._health_probe, None)

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        self.pipeline.set_state(Gst.State.PLAYING)
        self._glib_loop = GLib.MainLoop()
        self._loop_thread = threading.Thread(
            target=self._glib_loop.run, name="deepstream-glib-loop", daemon=True
        )
        self._loop_thread.start()
        logger.info(
            f"DeepStream pipeline started (batch-size={MUX_BATCH_SIZE}, "
            f"{MUX_WIDTH}x{MUX_HEIGHT})"
        )

    # ── Source add/remove — the standard DeepStream "runtime source" pattern ──

    def start_stream(self, camera_id: str, rtsp_url: str) -> bool:
        if not _DEEPSTREAM_AVAILABLE:
            with self._lock:
                self._stub_health[camera_id] = "UP"
            logger.info(f"[stub] Stream STARTED: {camera_id} → {rtsp_url}")
            return True

        with self._lock:
            if camera_id in self._sources:
                logger.warning(f"Camera {camera_id} already active — replacing")
                self._remove_source_locked(camera_id)
            pad_index = self._next_pad_index
            self._next_pad_index += 1

        source_bin = self._make_source_bin(camera_id, rtsp_url, pad_index)
        self.pipeline.add(source_bin)

        sinkpad = self.streammux.get_request_pad(f"sink_{pad_index}")
        srcpad  = source_bin.get_static_pad("src")
        if not sinkpad or not srcpad:
            logger.error(f"[{camera_id}] Could not link source bin into streammux")
            return False
        srcpad.link(sinkpad)
        source_bin.sync_state_with_parent()

        with self._lock:
            self._sources[camera_id] = _SourceState(camera_id, rtsp_url, source_bin, pad_index)

        if self._on_source_added:
            try:
                self._on_source_added(self.pipeline, self.streammux, pad_index)
            except Exception:
                logger.exception(f"[{camera_id}] on_source_added callback failed")

        logger.info(f"[{camera_id}] Source STARTED (source_id={pad_index}): {rtsp_url}")
        return True

    def stop_stream(self, camera_id: str) -> None:
        if not _DEEPSTREAM_AVAILABLE:
            with self._lock:
                self._stub_health.pop(camera_id, None)
            logger.info(f"[stub] Stream STOPPED: {camera_id}")
            return

        with self._lock:
            self._remove_source_locked(camera_id)

    def _remove_source_locked(self, camera_id: str) -> None:
        """
        Graceful runtime removal: send EOS down this source's pad and let a
        probe on the sink pad catch it, then unlink/release/remove on the
        main GLib thread. Must not block the pipeline's streaming thread —
        this schedules the actual teardown via GLib.idle_add.
        """
        state = self._sources.get(camera_id)
        if not state:
            logger.warning(f"Stop requested for unknown camera: {camera_id}")
            return
        if state.removing:
            return
        state.removing = True

        sinkpad = self.streammux.get_static_pad(f"sink_{state.pad_index}")
        if sinkpad is None:
            self._sources.pop(camera_id, None)
            return

        def _send_eos_and_finish():
            sinkpad.send_event(Gst.Event.new_eos())
            GLib.timeout_add(200, lambda: self._finish_removal(camera_id) and False)
            return False

        GLib.idle_add(_send_eos_and_finish)

    def _finish_removal(self, camera_id: str) -> bool:
        with self._lock:
            state = self._sources.pop(camera_id, None)
        if not state:
            return True
        sinkpad = self.streammux.get_static_pad(f"sink_{state.pad_index}")
        if sinkpad:
            self.streammux.release_request_pad(sinkpad)
        state.bin.set_state(Gst.State.NULL)
        self.pipeline.remove(state.bin)
        logger.info(f"[{camera_id}] Source STOPPED and removed")
        return True

    def _make_source_bin(self, camera_id: str, rtsp_url: str, pad_index: int):
        """uridecodebin wrapped in a Gst.Bin, with a ghost pad exposed once
        the decoder negotiates caps (pad-added is emitted asynchronously)."""
        bin_ = Gst.Bin.new(f"source-bin-{pad_index}")
        decoder = Gst.ElementFactory.make("uridecodebin", f"decoder-{pad_index}")
        decoder.set_property("uri", rtsp_url)

        def _on_pad_added(_element, pad, *_args):
            caps = pad.get_current_caps() or pad.query_caps(None)
            if not caps or not caps.get_structure(0).get_name().startswith("video"):
                return
            ghost_pad = bin_.get_static_pad("src")
            if ghost_pad and not ghost_pad.is_linked():
                pad.link(ghost_pad.get_target() or ghost_pad)

        def _on_child_added(_element, obj, name, *_args):
            # RTSP-specific tuning goes here if needed, e.g.:
            # if "rtspsrc" in name: obj.set_property("latency", 200)
            pass

        decoder.connect("pad-added", _on_pad_added)
        decoder.connect("deep-element-added", _on_child_added)
        bin_.add(decoder)

        # Ghost pad added now, target linked once uridecodebin exposes its
        # real pad above — standard deferred-pad pattern for uridecodebin.
        ghost = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
        bin_.add_pad(ghost)
        return bin_

    # ── Health — bus messages (hard failures) ────────────────────────────────

    def _on_bus_message(self, _bus, message, *_args):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            camera_id = self._camera_id_for_element(message.src)
            logger.error(f"[{camera_id or 'pipeline'}] GStreamer ERROR: {err} ({debug})")
            if camera_id:
                self._mark_failure(camera_id)
        elif t == Gst.MessageType.EOS:
            camera_id = self._camera_id_for_element(message.src)
            if camera_id:
                logger.warning(f"[{camera_id}] EOS received")
                self._mark_failure(camera_id)
        return True

    def _camera_id_for_element(self, element) -> Optional[str]:
        with self._lock:
            for cid, state in self._sources.items():
                if element == state.bin or (element and element.get_parent() == state.bin):
                    return cid
        return None

    def _mark_failure(self, camera_id: str) -> None:
        with self._lock:
            state = self._sources.get(camera_id)
            if state:
                state.consecutive_failures += 1

    # ── Health — pad probe (frame staleness, independent of AI inference) ────

    def _health_probe(self, _pad, info, _user_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        now = time.monotonic()
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        seen_source_ids = set()
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            seen_source_ids.add(frame_meta.source_id)
            l_frame = l_frame.next

        with self._lock:
            for state in self._sources.values():
                if state.pad_index in seen_source_ids:
                    state.last_seen_at = now
                    state.consecutive_failures = 0

        return Gst.PadProbeReturn.OK

    def health(self, state: "_SourceState", now: float) -> str:
        if state.consecutive_failures >= CAMERA_FAIL_THRESHOLD:
            return "DOWN"
        if state.last_seen_at is None:
            return "STARTING"
        if now - state.last_seen_at > CAMERA_STALE_AFTER_SECONDS:
            return "DOWN"
        return "UP"

    # ── Public read API — matches CameraStreamManager exactly ────────────────

    def get_camera_count(self) -> int:
        if not _DEEPSTREAM_AVAILABLE:
            with self._lock:
                return len(self._stub_health)
        with self._lock:
            return len(self._sources)

    def get_active_camera_ids(self) -> List[str]:
        if not _DEEPSTREAM_AVAILABLE:
            with self._lock:
                return list(self._stub_health.keys())
        with self._lock:
            return list(self._sources.keys())

    def get_average_fps(self) -> Optional[float]:
        """
        Informational only — never used for health/DEGRADED decisions.
        DeepStream doesn't give a simple per-camera decode FPS the way
        OpenCV's manual loop timing does; wire this up to your pipeline's
        own FPS-probe/perf-measurement element if you want a real number.
        Returns None (no cameras) rather than a fabricated value.
        """
        count = self.get_camera_count()
        return None if count == 0 else None

    def get_total_errors(self) -> int:
        """Informational only — never used for health/DEGRADED decisions."""
        if not _DEEPSTREAM_AVAILABLE:
            return 0
        with self._lock:
            return sum(s.consecutive_failures for s in self._sources.values())

    def get_camera_health(self) -> Dict[str, str]:
        """{camera_id: 'STARTING'|'UP'|'DOWN'} — sent to the central server
        every heartbeat. Same contract as camera_stream_manager.py."""
        if not _DEEPSTREAM_AVAILABLE:
            with self._lock:
                return dict(self._stub_health)
        now = time.monotonic()
        with self._lock:
            return {cid: self.health(state, now) for cid, state in self._sources.items()}

    def get_all(self) -> dict:
        if not _DEEPSTREAM_AVAILABLE:
            with self._lock:
                return {cid: {"health": h} for cid, h in self._stub_health.items()}
        now = time.monotonic()
        with self._lock:
            return {
                cid: {
                    "health": self.health(state, now),
                    "last_frame_age_seconds": (
                        round(now - state.last_seen_at, 1)
                        if state.last_seen_at is not None else None
                    ),
                    "source_id": state.pad_index,
                }
                for cid, state in self._sources.items()
            }


if __name__ == "__main__":
    # Minimal smoke test — attach a trivial inference branch (fakesink only)
    # so the pipeline is runnable standalone without a real model configured.
    # Replace on_source_added with your real nvinfer/tracker/OSD/sink chain.
    logging.basicConfig(level=logging.INFO)

    def _attach_fakesink(pipeline, streammux, source_id):
        convert = Gst.ElementFactory.make("nvvideoconvert", f"conv-{source_id}")
        sink    = Gst.ElementFactory.make("fakesink", f"sink-{source_id}")
        pipeline.add(convert)
        pipeline.add(sink)
        convert.link(sink)
        streammux.get_static_pad("src").link(convert.get_static_pad("sink"))
        convert.sync_state_with_parent()
        sink.sync_state_with_parent()

    mgr = DeepStreamStreamManager(on_source_added=_attach_fakesink)
    mgr.start_stream("cam-test-1", "rtsp://127.0.0.1:8554/test")
    time.sleep(60)
    print(mgr.get_camera_health())
