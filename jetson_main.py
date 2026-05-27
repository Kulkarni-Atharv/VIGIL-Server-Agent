"""
VIGIL Jetson Main Entry Point
Starts both the command receiver (FastAPI on port 8001) and the
telemetry heartbeat loop in the same process so they share one
CameraStreamManager instance via shared_state.py.

Usage:
    python jetson_main.py
"""

import asyncio
import logging
import os
import threading
import time

import uvicorn
from dotenv import load_dotenv

load_dotenv()

SERVER_URL          = os.getenv("SERVER_URL")
JETSON_ID           = os.getenv("JETSON_ID")
API_KEY             = os.getenv("API_KEY")
HEARTBEAT_INTERVAL  = int(os.getenv("HEARTBEAT_INTERVAL", "2"))
MOCK_JETSON_METRICS = os.getenv("MOCK_JETSON_METRICS", "false").lower() == "true"
CMD_RECEIVER_PORT   = int(os.getenv("CMD_RECEIVER_PORT", "8001"))

for _var in ("SERVER_URL", "JETSON_ID", "API_KEY"):
    if not os.getenv(_var):
        raise SystemExit(f"ERROR: '{_var}' is not set — add it to your .env file")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_command_receiver():
    """Run the FastAPI command receiver in a background thread."""
    from command_receiver import app
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=CMD_RECEIVER_PORT,
        log_level="warning",   # suppress uvicorn's own verbose output
    )


def run_heartbeat_loop():
    """Send telemetry heartbeats to the central server in a loop."""
    import socket
    import requests
    import psutil
    import re
    import subprocess
    from typing import Optional, Tuple
    from shared_state import stream_mgr

    def get_local_ip() -> str:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"

    def get_tegrastats_metrics() -> Tuple[Optional[float], Optional[float]]:
        try:
            proc = subprocess.Popen(
                ["tegrastats", "--interval", "500"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            line = proc.stdout.readline()
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
            if not line.strip():
                return None, None
            gpu_m = re.search(r"GR3D_FREQ\s+(\d+)%", line)
            gpu   = float(gpu_m.group(1)) if gpu_m else None
            temp_m = (
                re.search(r"gpu@([\d.]+)C", line, re.IGNORECASE) or
                re.search(r"tj@([\d.]+)C",  line, re.IGNORECASE) or
                re.search(r"cpu@([\d.]+)C", line, re.IGNORECASE)
            )
            temp = float(temp_m.group(1)) if temp_m else None
            return gpu, temp
        except FileNotFoundError:
            return None, None
        except Exception:
            return None, None

    def get_mock_metrics() -> Tuple[Optional[float], Optional[float]]:
        import random
        return round(random.uniform(15.0, 75.0), 1), round(random.uniform(36.0, 65.0), 1)

    logger.info(
        f"Heartbeat loop starting | ID:{JETSON_ID} | "
        f"Server:{SERVER_URL} | Interval:{HEARTBEAT_INTERVAL}s"
    )

    url     = f"{SERVER_URL}/api/v1/heartbeat"
    headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}

    while True:
        try:
            cpu  = psutil.cpu_percent(interval=1)
            ram  = psutil.virtual_memory().percent
            disk = psutil.disk_usage("/").percent

            if MOCK_JETSON_METRICS:
                gpu, temp = get_mock_metrics()
            else:
                gpu, temp = get_tegrastats_metrics()

            payload = {
                "jetson_id":             JETSON_ID,
                "hostname":              socket.gethostname(),
                "ip":                    get_local_ip(),
                "cpu_percent":           cpu,
                "ram_percent":           ram,
                "disk_percent":          disk,
                "gpu_percent":           gpu,
                "temperature":           temp,
                "assigned_camera_count": stream_mgr.get_camera_count(),
                "detection_fps":         stream_mgr.get_average_fps(),
                "rtsp_error_count":      stream_mgr.get_total_errors(),
            }

            resp   = requests.post(url, json=payload, headers=headers, timeout=5)
            resp.raise_for_status()
            result = resp.json()
            logger.info(
                f"Heartbeat OK | CPU:{cpu:.1f}% RAM:{ram:.1f}% "
                f"GPU:{gpu} Temp:{temp}°C "
                f"Cams:{stream_mgr.get_camera_count()} "
                f"→ {result.get('load_status')} (score {result.get('load_score')})"
            )

        except requests.exceptions.ConnectionError:
            logger.warning(f"Cannot reach server at {SERVER_URL} — will retry")
        except requests.exceptions.Timeout:
            logger.warning("Heartbeat timed out — will retry")
        except requests.exceptions.HTTPError as exc:
            logger.error(f"Server rejected heartbeat: {exc}")
        except Exception as exc:
            logger.error(f"Heartbeat error: {exc}")

        time.sleep(HEARTBEAT_INTERVAL)


if __name__ == "__main__":
    logger.info(f"VIGIL Jetson starting | ID:{JETSON_ID}")

    # Start command receiver in a background daemon thread
    cmd_thread = threading.Thread(target=run_command_receiver, daemon=True)
    cmd_thread.start()
    logger.info(f"Command receiver listening on port {CMD_RECEIVER_PORT}")

    # Run heartbeat loop in the main thread
    run_heartbeat_loop()
