from __future__ import annotations

from functools import lru_cache

from redis import Redis

from app.config import REDIS_DB, REDIS_HOST, REDIS_PORT


@lru_cache(maxsize=1)
def get_redis() -> Redis:
    return Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=10,
        health_check_interval=30,
    )
