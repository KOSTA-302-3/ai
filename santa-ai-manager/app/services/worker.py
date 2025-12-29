# app/services/worker.py

import asyncio
import json
import logging
from app.core.config import settings
from app.core.connections import redis_client
from app.services.modal_service import trigger_inference
from app.services.centroid_service import CentroidService

logger = logging.getLogger(__name__)
centroid_service = CentroidService()

async def start_worker():
    logger.info(f"Worker 시작: [Inference: {settings.REDIS_QUEUE_NAME}, Feedback: {settings.REDIS_FEEDBACK_QUEUE_NAME}]")
    
    await asyncio.gather(
        watch_inference_queue(),
        watch_feedback_queue()
    )

async def watch_inference_queue():
    """기존: 추론 요청 처리"""
    logger.info("Inference Queue 감시 시작...")
    while True:
        try:
            job = await redis_client.blpop(settings.REDIS_QUEUE_NAME, timeout=1)
            if job:
                job_info = json.loads(job[1])
                logger.info(f"[Inference] 작업 수신: {job_info.get('job_id')}")
                asyncio.create_task(trigger_inference(job_info))
        except Exception as e:
            logger.error(f"[Inference] 에러: {e}")
            await asyncio.sleep(1)
        await asyncio.sleep(0.01)

async def watch_feedback_queue():
    """신규: 피드백 반영 및 전체 레벨 재조정"""
    logger.info("Feedback Queue 감시 시작...")
    while True:
        try:
            job = await redis_client.blpop(settings.REDIS_FEEDBACK_QUEUE_NAME, timeout=1)
            if job:
                feedback_info = json.loads(job[1])
                logger.info(f"[Feedback] 피드백 수신: Post {feedback_info.get('post_id')} -> Level {feedback_info.get('correct_level')}")
                
                await centroid_service.process_feedback_job(feedback_info)
                
        except Exception as e:
            logger.error(f"[Feedback] 에러: {e}")
            await asyncio.sleep(1)
        await asyncio.sleep(0.01)