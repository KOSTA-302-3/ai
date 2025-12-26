from redis.asyncio import Redis
from app.core.config import settings

redis_url = f"rediss://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}"

redis_client = Redis.from_url(
    redis_url,
    decode_responses=True,
    ssl_cert_reqs=None
)