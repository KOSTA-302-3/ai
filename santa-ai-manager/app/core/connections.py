from redis.asyncio import Redis, ConnectionPool
from app.core.config import settings
import ssl

# 연결 풀 생성
pool = ConnectionPool(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    password=settings.REDIS_PASSWORD,
    decode_responses=True,
    ssl=True,               
    ssl_cert_reqs=None,      
    socket_timeout=5.0,       
    socket_keepalive=True     
)

# 비동기 클라이언트 생성
redis_client = Redis(connection_pool=pool)
