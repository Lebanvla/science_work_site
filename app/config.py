from __future__ import annotations

import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

STORAGE_DIR = APP_DIR / "storage"
INPUT_DIR = STORAGE_DIR / "input"
OUTPUT_DIR = STORAGE_DIR / "output"
TEMPLATES_DIR = APP_DIR / "templates"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(25 * 1024 * 1024)))

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

QUEUE_NAME = os.getenv("QUEUE_NAME", "segmentation_queue")
RESULT_KEY_PREFIX = "result:"
RESULT_TTL_SECONDS = int(os.getenv("RESULT_TTL_SECONDS", "3600"))
CLEANUP_DELAY_SECONDS = int(os.getenv("CLEANUP_DELAY_SECONDS", "3600"))

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "nvidia/segformer-b0-finetuned-cityscapes-768-768",
)
ROAD_CLASS_NAME = os.getenv("ROAD_CLASS_NAME", "road")
ROAD_CLASS_INDEX = os.getenv("ROAD_CLASS_INDEX")


def ensure_directories() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def result_key(picture_hash: str) -> str:
    return f"{RESULT_KEY_PREFIX}{picture_hash}"
