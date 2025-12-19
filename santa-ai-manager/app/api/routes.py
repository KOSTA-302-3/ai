# app/api/routes.py
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from app.core.connections import redis_client
from app.core.config import settings
import json
import logging

# 로그 설정
logger = logging.getLogger(__name__)
router = APIRouter()

# Modal로부터 받을 결과 데이터 규격 정의
class InferenceResult(BaseModel):
    job_id: str
    unified_vector: list  
    status: str

@router.post("/inference-result")
async def receive_result(result: InferenceResult, x_santa_token: str = Header(None)):

    
    # 보안 검사: 헤더의 토큰과 설정된 보안 토큰이 일치하는지 확인
    if x_santa_token != settings.SANTA_SECRET_TOKEN:
        logger.warning(f"승인되지 않은 접근 시도! Job ID: {result.job_id}")
        raise HTTPException(status_code=403, detail="Unauthorized")

    logger.info(f"[Webhook] 결과 수신 완료! Job ID: {result.job_id}")
    
    # Redis에 결과 및 상태 저장
    try:
        await redis_client.hset(
            f"job:{result.job_id}", 
            mapping={
                "unified_vector": json.dumps(result.unified_vector),
                "status": "completed"
            }
        )
        logger.info(f"Job {result.job_id} 상태 업데이트 및 벡터 저장 완료")
    except Exception as e:
        logger.error(f"Redis 저장 중 에러 발생: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
    return {"status": "success"}