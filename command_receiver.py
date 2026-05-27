"""
VIGIL Jetson Command Receiver
Runs as a FastAPI server on port 8001.
Receives ASSIGN_CAMERA / REMOVE_CAMERA / GET_STATUS commands
from the Central Server and delegates to the CameraStreamManager.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from shared_state import stream_mgr

load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise SystemExit("ERROR: 'API_KEY' is not set in .env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="VIGIL Jetson Command Receiver", version="2.0.0")


class Command(BaseModel):
    command:   str
    camera_id: Optional[str] = None
    rtsp_url:  Optional[str] = None


@app.post("/api/v1/command")
async def receive_command(cmd: Command, x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    logger.info(f"Command received: {cmd.command} | camera={cmd.camera_id}")

    if cmd.command == "ASSIGN_CAMERA":
        if not cmd.camera_id or not cmd.rtsp_url:
            raise HTTPException(
                status_code=422, detail="camera_id and rtsp_url are required"
            )
        success = stream_mgr.start_stream(cmd.camera_id, cmd.rtsp_url)
        return {
            "status":    "ACKNOWLEDGED" if success else "FAILED",
            "camera_id": cmd.camera_id,
        }

    elif cmd.command == "REMOVE_CAMERA":
        if not cmd.camera_id:
            raise HTTPException(status_code=422, detail="camera_id is required")
        stream_mgr.stop_stream(cmd.camera_id)
        return {"status": "STOPPED", "camera_id": cmd.camera_id}

    elif cmd.command == "GET_STATUS":
        return {
            "status":       "ACKNOWLEDGED",
            "camera_count": stream_mgr.get_camera_count(),
            "streams":      stream_mgr.get_all(),
        }

    else:
        raise HTTPException(status_code=400, detail=f"Unknown command: {cmd.command}")
