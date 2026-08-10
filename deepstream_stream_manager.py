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

── Inference branch — nvinfer / nvtracker / nvdsosd / sink ───────────────────
Built ONCE, right after nvstreammux, not per-camera — inference runs on the
whole batch (all cameras together), which is the point of batching in the
first place. Wired up automatically if DS_INFER_CONFIG_PATH is set in
.env, pointing at your existing nvinfer config file (the same one your
current standalone DeepStream pipeline already uses). Optional nvtracker
(DS_TRACKER_CONFIG_PATH) and nvdsosd (DS_ENABLE_OSD) stages, and a
configurable sink (DS_SINK_TYPE: fakesink/display/rtsp/file).

If DS_INFER_CONFIG_PATH is unset, no inference branch is built at all —
the pipeline still runs and reports camera health (ingestion + health is
useful on its own for testing), it just doesn't detect anything. This is
the safe default until you're ready to point it at a real model config.

Detections are extracted via a probe after nvinfer (see
_detection_probe) and currently just logged, throttled per source — this
is the extension point for wiring detections into an alerting/event
pipeline later (e.g. POSTing to the central server's /api/v1/events).
Nothing here decides what "happens" with a detection beyond logging it,
since that's a product decision this module doesn't own.

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

# ── Inference branch config — point these at your existing model files ───────
# Path to the nvinfer config .txt your current standalone DeepStream pipeline
# already uses (model engine path, labels file, etc. all live inside that
# file, same as any standard DeepStream app). Unset = no inference branch;
# the pipeline still ingests + health-checks cameras, just doesn't detect.
DS_INFER_CONFIG_PATH   = os.getenv("DS_INFER_CONFIG_PATH", "").strip()
# Optional — path to an nvtracker config (e.g. NvDCF/IOU tracker config file).
DS_TRACKER_CONFIG_PATH = os.getenv("DS_TRACKER_CONFIG_PATH", "").strip()
# Optional — nvdsosd draws bounding boxes; only useful if you're also using
# a visual sink (display/rtsp/file). No effect on detection/health logic.
DS_ENABLE_OSD          = os.getenv("DS_ENABLE_OSD", "false").strip().lower() == "true"
# fakesink (default, headless — detections are still extracted via the probe
# even though nothing is rendered) | display (nveglglessink, needs a screen) |
# file (mp4 via nvv4l2h264enc, DS_SINK_FILE_PATH) | rtsp (basic RTSP re-stream
# out, DS_SINK_RTSP_PORT).
DS_SINK_TYPE           = os.getenv("DS_SINK_TYPE", "fakesink").strip().lower()
DS_SINK_FILE_PATH      = os.getenv("DS_SINK_FILE_PATH", "/tmp/vigil_deepstream_out.mp4")
DS_SINK_RTSP_PORT      = int(os.getenv("DS_SINK_RTSP_PORT", "8554"))
# Minimum seconds between detection log lines PER SOURCE — inference runs on
# every batch (many times a second); this just throttles logging, not
# detection itself.
DS_DETECTION_LOG_INTERVAL_SECONDS = float(os.getenv("DS_DETECTION_LOG_INTERVAL_SECONDS", "5"))


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
        on_source_added(pipeline, streammux, source_id): OPTIONAL — called
        the first time a source's pad is linked into streammux. Most
        DeepStream apps don't need this: inference is built ONCE, shared
        across all sources (see _build_inference_branch), not per-camera.
        Use this hook only for genuinely per-source needs (e.g. a
        per-camera output sink) beyond the shared inference branch below.
        """
        self._lock = threading.Lock()
        self._sources: Dict[str, _SourceState] = {}
        self._next_pad_index = 0
        self._on_source_added = on_source_added
        self._loop_thread: Optional[threading.Thread] = None
        self._glib_loop = None
        self._last_detection_log: Dict[int, float] = {}   # source_id -> monotonic time

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

        # Inference branch — built once, shared across every camera. No-op
        # (pipeline still runs, still reports health) if DS_INFER_CONFIG_PATH
        # isn't set. See module docstring.
        self._build_inference_branch()

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        self.pipeline.set_state(Gst.State.PLAYING)
        self._glib_loop = GLib.MainLoop()
        self._loop_thread = threading.Thread(
            target=self._glib_loop.run, name="deepstream-glib-loop", daemon=True
        )
        self._loop_thread.start()
        if DS_INFER_CONFIG_PATH:
            logger.info(f"DeepStream inference branch active — config: {DS_INFER_CONFIG_PATH}")
        else:
            logger.warning(
                "DS_INFER_CONFIG_PATH not set — pipeline will ingest and "
                "health-check cameras but will NOT run detection. Set it in "
                ".env once you're ready to point this at your real model."
            )

    # ── Inference branch — nvinfer / nvtracker / nvdsosd / sink, built once ──

    def _build_inference_branch(self) -> None:
        if not DS_INFER_CONFIG_PATH:
            return
        if not os.path.isfile(DS_INFER_CONFIG_PATH):
            logger.error(
                f"DS_INFER_CONFIG_PATH='{DS_INFER_CONFIG_PATH}' does not exist "
                f"— skipping inference branch, pipeline will only ingest/health-check"
            )
            return

        pgie = Gst.ElementFactory.make("nvinfer", "vigil-pgie")
        pgie.set_property("config-file-path", DS_INFER_CONFIG_PATH)
        self.pipeline.add(pgie)
        self.streammux.link(pgie)
        last = pgie

        if DS_TRACKER_CONFIG_PATH:
            if os.path.isfile(DS_TRACKER_CONFIG_PATH):
                tracker = Gst.ElementFactory.make("nvtracker", "vigil-tracker")
                self._apply_tracker_config(tracker, DS_TRACKER_CONFIG_PATH)
                self.pipeline.add(tracker)
                last.link(tracker)
                last = tracker
            else:
                logger.error(
                    f"DS_TRACKER_CONFIG_PATH='{DS_TRACKER_CONFIG_PATH}' does not "
                    f"exist — continuing without a tracker"
                )

        # Detection probe — reads NvDsObjectMeta off every batch right after
        # inference (and tracking, if configured). This is the extension
        # point for turning detections into alerts/events later.
        last.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, self._detection_probe, None
        )

        convert = Gst.ElementFactory.make("nvvideoconvert", "vigil-convert")
        self.pipeline.add(convert)
        last.link(convert)
        last = convert

        if DS_ENABLE_OSD:
            osd = Gst.ElementFactory.make("nvdsosd", "vigil-osd")
            self.pipeline.add(osd)
            last.link(osd)
            last = osd

        sink = self._build_sink()
        self.pipeline.add(sink)
        last.link(sink)

    def _apply_tracker_config(self, tracker, config_path: str) -> None:
        """nvtracker takes its settings from a key=value config file, not a
        single config-file-path property — parse the [tracker] section the
        same way NVIDIA's own reference apps do."""
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(config_path)
        if "tracker" not in cfg:
            logger.error(f"No [tracker] section in {config_path} — using nvtracker defaults")
            return
        section = cfg["tracker"]
        prop_map = {
            "tracker-width":            ("tracker-width", int),
            "tracker-height":           ("tracker-height", int),
            "gpu-id":                   ("gpu-id", int),
            "ll-lib-file":              ("ll-lib-file", str),
            "ll-config-file":           ("ll-config-file", str),
        }
        for key, (prop, cast) in prop_map.items():
            if key in section:
                try:
                    tracker.set_property(prop, cast(section[key]))
                except Exception:
                    logger.exception(f"Failed to set nvtracker property {prop}")

    def _build_sink(self):
        """Build the tail of the pipeline per DS_SINK_TYPE. fakesink
        (default) is the right choice for a headless server agent — you
        still get detections via the probe above even though nothing is
        rendered or written anywhere."""
        if DS_SINK_TYPE == "fakesink":
            sink = Gst.ElementFactory.make("fakesink", "vigil-sink")
            sink.set_property("sync", False)
            return sink

        if DS_SINK_TYPE == "display":
            sink = Gst.ElementFactory.make("nveglglessink", "vigil-sink")
            sink.set_property("sync", False)
            return sink

        if DS_SINK_TYPE == "file":
            bin_ = Gst.Bin.new("vigil-file-sink-bin")
            enc     = Gst.ElementFactory.make("nvv4l2h264enc", "vigil-enc")
            parse   = Gst.ElementFactory.make("h264parse", "vigil-parse")
            mux     = Gst.ElementFactory.make("qtmux", "vigil-mux")
            filesink = Gst.ElementFactory.make("filesink", "vigil-filesink")
            filesink.set_property("location", DS_SINK_FILE_PATH)
            for el in (enc, parse, mux, filesink):
                bin_.add(el)
            enc.link(parse)
            parse.link(mux)
            mux.link(filesink)
            ghost = Gst.GhostPad.new("sink", enc.get_static_pad("sink"))
            bin_.add_pad(ghost)
            return bin_

        if DS_SINK_TYPE == "rtsp":
            # Basic RTP/UDP H264 output, viewable with e.g.
            # `ffplay udp://<jetson-ip>:<DS_SINK_RTSP_PORT>` — NOT a full
            # mountable RTSP server (that needs GstRtspServer, more setup
            # than fits here). Good enough to visually spot-check detections
            # on the network; swap for your own sink if you need a real
            # RTSP mount point.
            bin_ = Gst.Bin.new("vigil-rtsp-sink-bin")
            enc  = Gst.ElementFactory.make("nvv4l2h264enc", "vigil-enc")
            pay  = Gst.ElementFactory.make("rtph264pay", "vigil-pay")
            sink = Gst.ElementFactory.make("udpsink", "vigil-udpsink")
            sink.set_property("port", DS_SINK_RTSP_PORT)
            sink.set_property("sync", False)
            for el in (enc, pay, sink):
                bin_.add(el)
            enc.link(pay)
            pay.link(sink)
            ghost = Gst.GhostPad.new("sink", enc.get_static_pad("sink"))
            bin_.add_pad(ghost)
            return bin_

        logger.error(f"Unknown DS_SINK_TYPE='{DS_SINK_TYPE}' — falling back to fakesink")
        sink = Gst.ElementFactory.make("fakesink", "vigil-sink")
        sink.set_property("sync", False)
        return sink

    def _detection_probe(self, _pad, info, _user_data):
        """
        Extracts detections per batched buffer and logs them, throttled per
        source. This is the extension point for wiring detections into an
        alerting/event pipeline (e.g. POSTing to the central server) —
        nothing here decides what should happen with a detection beyond
        logging it.
        """
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        now = time.monotonic()
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            source_id  = frame_meta.source_id

            objects = []
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                objects.append((obj_meta.obj_label, round(obj_meta.confidence, 2)))
                l_obj = l_obj.next

            if objects:
                last_log = self._last_detection_log.get(source_id, 0)
                if now - last_log >= DS_DETECTION_LOG_INTERVAL_SECONDS:
                    camera_id = self._camera_id_for_source(source_id)
                    logger.info(
                        f"[{camera_id or source_id}] Detected: "
                        + ", ".join(f"{label}({conf})" for label, conf in objects)
                    )
                    self._last_detection_log[source_id] = now

            l_frame = l_frame.next

        return Gst.PadProbeReturn.OK

    def _camera_id_for_source(self, source_id: int) -> Optional[str]:
        with self._lock:
            for cid, state in self._sources.items():
                if state.pad_index == source_id:
                    return cid
        return None
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
    # Minimal smoke test. The inference branch (nvinfer/tracker/OSD/sink) is
    # built automatically from DS_INFER_CONFIG_PATH etc. in .env — nothing
    # extra to wire up here. Without DS_INFER_CONFIG_PATH set, this just
    # exercises ingestion + health tracking (no detection), which is a fine
    # first smoke test before pointing it at a real model.
    logging.basicConfig(level=logging.INFO)

    mgr = DeepStreamStreamManager()
    mgr.start_stream("cam-test-1", "rtsp://127.0.0.1:8554/test")
    time.sleep(60)
    print(mgr.get_camera_health())
