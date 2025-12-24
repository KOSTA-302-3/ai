# app/api/routes.py
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from app.core.connections import redis_client
from app.core.config import settings
import json
import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import os

# [Database]
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.post import Post

logger = logging.getLogger(__name__)
router = APIRouter()

QDRANT_HOST = settings.QDRANT_HOST
QDRANT_PORT = settings.QDRANT_PORT

class InferenceResult(BaseModel):
    job_id: int
    unified_vector: list  
    status: str

@router.post("/inference-result")
async def receive_result(
    result: InferenceResult, 
    x_santa_token: str = Header(None),
    db: Session = Depends(get_db) # ✅ DB 세션 주입
):

    if x_santa_token != settings.SANTA_SECRET_TOKEN:
        logger.warning(f"승인되지 않은 접근 시도! Post ID: {result.job_id}")
        raise HTTPException(status_code=403, detail="Unauthorized")

    logger.info(f"[Webhook] 결과 수신: Post ID {result.job_id}")
    
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        client.upsert(
            collection_name="santa_images",
            points=[
                PointStruct(
                    id=result.job_id,
                    vector=result.unified_vector,
                    payload={
                        "post_id": result.job_id,
                        "status": "completed"
                    }
                )
            ]
        )
        logger.info(f"✅ Qdrant 저장 완료 (ID: {result.job_id})")

        target_post = db.query(Post).filter(Post.post_id == result.job_id).first()
        
        if target_post:
            target_post.level = 1
            db.commit()
            logger.info(f"✅ RDS Level 업데이트 완료 (Post ID: {result.job_id})")
        else:
            logger.warning(f"⚠️ RDS에서 해당 게시물을 찾을 수 없음 (ID: {result.job_id})")

    except Exception as e:
        logger.error(f"처리 중 에러 발생: {e}")
        db.rollback() # DB 에러 시 롤백
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
