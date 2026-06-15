from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path

from redis import RedisError

from app.config import (
    CLEANUP_DELAY_SECONDS,
    MODEL_NAME,
    OUTPUT_DIR,
    QUEUE_NAME,
    RESULT_TTL_SECONDS,
    ensure_directories,
    result_key,
)
from app.model_runner import load_model, process_image
from app.redis_client import get_redis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def schedule_file_cleanup(input_path: str | Path, output_path: str | Path, delay_seconds: int = 3600) -> None:
    def cleanup() -> None:
        for path in (Path(input_path), Path(output_path)):
            try:
                path.unlink(missing_ok=True)
                logger.info("Cleanup completed for %s", path)
            except OSError:
                logger.exception("Failed to cleanup file %s", path)

    timer = threading.Timer(delay_seconds, cleanup)
    timer.daemon = True
    timer.start()


def _build_output_hash(picture_hash: str) -> str:
    raw = f"{picture_hash}:{MODEL_NAME}:{time.time_ns()}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_processing_error(picture_hash: str, message: str) -> None:
    redis = get_redis()
    redis.set(
        result_key(picture_hash),
        json.dumps({"status": "error", "message": message[:500]}),
        ex=RESULT_TTL_SECONDS,
    )


def handle_task(task_payload: str) -> None:
    input_path: Path | None = None
    output_path: Path | None = None
    picture_hash = "unknown"

    try:
        task = json.loads(task_payload)
        picture_hash = str(task["picture_hash"])
        input_path = Path(task["input_path"])

        if not input_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_path}")

        output_hash = _build_output_hash(picture_hash)
        output_path = OUTPUT_DIR / f"{output_hash}.png"

        logger.info("Received task picture_hash=%s", picture_hash)
        logger.info("Input path: %s", input_path)
        logger.info("Output path: %s", output_path)

        process_image(str(input_path), str(output_path))

        redis = get_redis()
        redis.set(result_key(picture_hash), output_hash, ex=RESULT_TTL_SECONDS)
        schedule_file_cleanup(input_path, output_path, delay_seconds=CLEANUP_DELAY_SECONDS)

        logger.info("Task completed picture_hash=%s output_hash=%s", picture_hash, output_hash)

    except Exception as exc:
        logger.exception("Failed to process task picture_hash=%s", picture_hash)
        if picture_hash != "unknown":
            try:
                _write_processing_error(picture_hash, str(exc))
            except RedisError:
                logger.exception("Failed to write processing error to Redis")
        if input_path is not None:
            schedule_file_cleanup(
                input_path,
                output_path or OUTPUT_DIR / f"failed-{os.getpid()}.png",
                delay_seconds=CLEANUP_DELAY_SECONDS,
            )


def main() -> None:
    ensure_directories()

    logger.info("Worker starting")
    logger.info("Loading model once at worker startup")
    load_model()
    logger.info("Model is ready")

    redis = get_redis()
    logger.info("Listening Redis queue: %s", QUEUE_NAME)

    while True:
        try:
            item = redis.blpop(QUEUE_NAME, timeout=5)
            if item is None:
                continue
            _, task_payload = item
            handle_task(task_payload)
        except RedisError:
            logger.exception("Redis error while reading queue")
        except KeyboardInterrupt:
            logger.info("Worker stopped by user")
            break


if __name__ == "__main__":
    main()
