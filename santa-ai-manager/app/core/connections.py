from redis.asyncio import Redis, ConnectionPool # asyncio용으로 변경
from app.core.config import settings

pool = ConnectionPool(
    host=settings.REDIS_HOST, 
    port=settings.REDIS_PORT, 
    decode_responses=True
)

redis_client = Redis(connection_pool=pool)