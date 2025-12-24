# app/api/routes.py
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from app.core.connections import redis_client
from app.core.config import settings
import json
import logging

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import os

logger = logging.getLogger(__name__)
router = APIRouter()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

class InferenceResult(BaseModel):
    job_id: str
    unified_vector: list  
    status: str

@router.get("/internal/test-redis")
async def test_redis():
    try:
        # Redis에 ping을 날려 연결 확인
        pong = await redis_client.ping()
        return {"redis_status": "connected" if pong else "failed"}
    except Exception as e:
        return {"redis_status": "error", "detail": str(e)}

@router.post("/inference-result")
async def receive_result(result: InferenceResult, x_santa_token: str = Header(None)):

    if x_santa_token != settings.SANTA_SECRET_TOKEN:
        logger.warning(f"승인되지 않은 접근 시도! Job ID: {result.job_id}")
        raise HTTPException(status_code=403, detail="Unauthorized")

    logger.info(f"[Webhook] 결과 수신 완료! Job ID: {result.job_id}")
    
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

@router.post("/setup/qdrant")
async def setup_qdrant():
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # 컬렉션 이름 정의 (예: santa_images)
        collection_name = "santa_images"
        
        # 이미 존재하는지 확인
        collections = client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        
        if exists:
            return {"message": f"Collection '{collection_name}' already exists."}

        # 1152차원 SigLIP 벡터를 위한 컬렉션 생성
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1152,  # SigLIP-L/14 등 1152차원 모델 기준
                distance=Distance.COSINE # 이미지 유사도 검색에 최적화된 Cosine 유사도
            ),
        )
        return {"message": f"Collection '{collection_name}' created successfully!"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
