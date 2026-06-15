from __future__ import annotations

import hashlib
import json
import logging
import re
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
from redis import RedisError

from app.config import (
    ALLOWED_EXTENSIONS,
    INPUT_DIR,
    MAX_UPLOAD_SIZE_BYTES,
    OUTPUT_DIR,
    QUEUE_NAME,
    TEMPLATES_DIR,
    ensure_directories,
    result_key,
)
from app.redis_client import get_redis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

ensure_directories()

app = FastAPI(title="Road Segmentation App")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

_HEX_64_RE = re.compile(r"^[a-f0-9]{64}$")


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


def _validate_extension(filename: str | None) -> str:
    if not filename:
        raise HTTPException(status_code=400, detail="File name is required")

    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {allowed}")

    return extension


def _validate_image_bytes(content: bytes) -> None:
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="Uploaded file is too large")

    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    extension = _validate_extension(file.filename)
    content = await file.read()
    _validate_image_bytes(content)

    picture_hash = hashlib.sha256(content).hexdigest()
    input_path = INPUT_DIR / f"{picture_hash}{extension}"
    input_path.write_bytes(content)

    task = {
        "picture_hash": picture_hash,
        "input_path": str(input_path),
        "extension": extension,
    }

    try:
        redis = get_redis()
        redis.rpush(QUEUE_NAME, json.dumps(task))
    except RedisError as exc:
        logger.exception("Failed to enqueue task for picture_hash=%s", picture_hash)
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail="Redis queue is unavailable") from exc

    logger.info("Queued segmentation task for picture_hash=%s input_path=%s", picture_hash, input_path)
    return {"status": "queued", "picture_hash": picture_hash}


@app.get("/get/{picture_hash}")
def get_result(picture_hash: str):
    if not _HEX_64_RE.match(picture_hash):
        raise HTTPException(status_code=400, detail="Invalid picture_hash format")

    try:
        redis = get_redis()
        value = redis.get(result_key(picture_hash))
    except RedisError as exc:
        logger.exception("Failed to read result for picture_hash=%s", picture_hash)
        raise HTTPException(status_code=503, detail="Redis is unavailable") from exc

    if value is None:
        return {"status": "processing"}

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value

    if isinstance(parsed, dict):
        if parsed.get("status") == "error":
            return JSONResponse(
                status_code=200,
                content={
                    "status": "error",
                    "message": str(parsed.get("message", "Unknown processing error")),
                },
            )
        if "output_hash" in parsed:
            output_hash = str(parsed["output_hash"])
        else:
            return {"status": "processing"}
    else:
        output_hash = str(parsed)

    output_path = OUTPUT_DIR / f"{output_hash}.png"
    if not output_path.exists():
        return {
            "status": "error",
            "message": "Result metadata exists, but output file was not found. Please upload the image again.",
        }

    return {"status": "ready", "url": f"/static/output/{output_hash}.png"}
