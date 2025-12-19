# app/services/modal_service.py
import modal
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

async def trigger_inference(job_info):
    try:
        # Modal 앱 이름 'santa'에 등록된 'run_inference' 함수 로드
        f = modal.Function.from_name("santa", "run_inference")
        
        # 비동기로 실행을 던짐
        f.spawn(
            image_urls=job_info.get("image_urls"),
            content=job_info.get("content"),
            job_id=job_info.get("job_id"),
            callback_url=f"{settings.CALLBACK_BASE_URL}/internal/inference-result",
            secret_token=settings.SANTA_SECRET_TOKEN
        )
        logger.info(f"Modal 작업 요청 성공: {job_info.get('job_id')}")
        
    except Exception as e:
        logger.error(f"Modal 호출 중 에러 발생: {e}")