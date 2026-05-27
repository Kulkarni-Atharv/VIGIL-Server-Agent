"""
VIGIL Jetson Health Agent
Collects system metrics every HEARTBEAT_INTERVAL seconds and sends them
to the Central Server via a REST heartbeat endpoint.
"""

import logging
import os
import re
import socket
import subprocess
import time
from typing import Optional, Tuple

import psutil
import requests
from dotenv import load_dotenv

load_dotenv()

SERVER_URL          = os.getenv("SERVER_URL")
JETSON_ID           = os.getenv("JETSON_ID")
API_KEY             = os.getenv("API_KEY")
HEARTBEAT_INTERVAL  = int(os.getenv("HEARTBEAT_INTERVAL", "5"))
MOCK_JETSON_METRICS = os.getenv("MOCK_JETSON_METRICS", "false").lower() == "true"

for _var in ("SERVER_URL", "JETSON_ID", "API_KEY"):
    if not os.getenv(_var):
        raise SystemExit(f"ERROR: '{_var}' is not set — add it to your .env file")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Network helpers ───────────────────────────────────────────────────────────

def get_local_ip() -> str:
    """Return the primary outbound IP address of this machine."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


# ── Jetson-specific metrics via tegrastats ────────────────────────────────────

def parse_tegrastats_line(line: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract GR3D GPU usage and temperature from a single tegrastats output line.

    Example line (Jetson Nano / Xavier NX):
        RAM 1872/7764MB ... GR3D_FREQ 45%@921 ... CPU@42.5C GPU@40C
    """
    gpu_match = re.search(r"GR3D_FREQ\s+(\d+)%", line)
    gpu = float(gpu_match.group(1)) if gpu_match else None

    # Try gpu sensor → tj (junction, hottest point) → cpu — all case-insensitive
    temp_match = (
        re.search(r"gpu@([\d.]+)C", line, re.IGNORECASE) or
        re.search(r"tj@([\d.]+)C",  line, re.IGNORECASE) or
        re.search(r"cpu@([\d.]+)C", line, re.IGNORECASE)
    )
    temp = float(temp_match.group(1)) if temp_match else None

    return gpu, temp


def get_tegrastats_metrics() -> Tuple[Optional[float], Optional[float]]:
    """Run tegrastats, read one line, then terminate the process."""
    try:
        proc = subprocess.Popen(
            ["tegrastats", "--interval", "500"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        line = proc.stdout.readline()
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

        if not line.strip():
            return None, None

        return parse_tegrastats_line(line)

    except FileNotFoundError:
        logger.debug("tegrastats not found — GPU/temperature will be reported as null")
        return None, None
    except Exception as exc:
        logger.debug(f"tegrastats error: {exc}")
        return None, None


# ── Mock metrics for development / non-Jetson testing ────────────────────────

def get_mock_jetson_metrics() -> Tuple[Optional[float], Optional[float]]:
    """Return randomised GPU & temperature values for testing on PC."""
    import random
    gpu  = round(random.uniform(15.0, 75.0), 1)
    temp = round(random.uniform(36.0, 65.0), 1)
    return gpu, temp


# ── Metric collection ─────────────────────────────────────────────────────────

def collect_metrics() -> dict:
    cpu  = psutil.cpu_percent(interval=1)          # blocking 1-second sample
    ram  = psutil.virtual_memory().percent
    disk = psutil.disk_usage("/").percent

    if MOCK_JETSON_METRICS:
        gpu, temperature = get_mock_jetson_metrics()
    else:
        gpu, temperature = get_tegrastats_metrics()

    return {
        "jetson_id":    JETSON_ID,
        "hostname":     socket.gethostname(),
        "ip":           get_local_ip(),
        "cpu_percent":  cpu,
        "ram_percent":  ram,
        "disk_percent": disk,
        "gpu_percent":  gpu,
        "temperature":  temperature,
    }


# ── Heartbeat sender ──────────────────────────────────────────────────────────

def send_heartbeat(payload: dict) -> bool:
    url     = f"{SERVER_URL}/api/v1/heartbeat"
    headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=5)
        resp.raise_for_status()
        result = resp.json()
        logger.info(
            f"Heartbeat OK | "
            f"CPU:{payload['cpu_percent']:.1f}%  "
            f"RAM:{payload['ram_percent']:.1f}%  "
            f"GPU:{payload['gpu_percent']}  "
            f"Temp:{payload['temperature']}°C  "
            f"→ {result.get('load_status')}"
        )
        return True

    except requests.exceptions.ConnectionError:
        logger.warning(f"Cannot reach server at {SERVER_URL} — will retry in {HEARTBEAT_INTERVAL}s")
    except requests.exceptions.Timeout:
        logger.warning("Heartbeat request timed out — will retry")
    except requests.exceptions.HTTPError as exc:
        logger.error(f"Server rejected heartbeat: {exc}")
    except Exception as exc:
        logger.error(f"Unexpected error sending heartbeat: {exc}")
    return False


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logger.info(
        f"VIGIL Jetson Agent starting | "
        f"ID: {JETSON_ID} | Server: {SERVER_URL} | "
        f"Interval: {HEARTBEAT_INTERVAL}s"
    )
    if MOCK_JETSON_METRICS:
        logger.info("Mock Jetson metrics enabled (development/testing mode)")

    while True:
        try:
            payload = collect_metrics()
            send_heartbeat(payload)
        except Exception as exc:
            logger.error(f"Unhandled error in main loop: {exc}")

        time.sleep(HEARTBEAT_INTERVAL)


if __name__ == "__main__":
    main()
