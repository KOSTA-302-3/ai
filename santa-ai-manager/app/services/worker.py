# app/services/worker.py
import asyncio
import json
import logging
from app.core.config import settings # 설정 임포트
from app.core.connections import redis_client
from app.services.modal_service import trigger_inference

logger = logging.getLogger(__name__)

async def start_worker():
    logger.info(f"AI Manager 워커 감시 중인 큐: {settings.REDIS_QUEUE_NAME}")
    
    while True:
        try:
            job = await redis_client.blpop(settings.REDIS_QUEUE_NAME, timeout=1)
            
            if job:
                job_info = json.loads(job[1])
                logger.info(f"발견: {job_info.get('job_id')}")
                asyncio.create_task(trigger_inference(job_info))

        except Exception as e:
            logger.error(f"워커 에러: {e}")
            await asyncio.sleep(2)

        await asyncio.sleep(0.01)