from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, Header, Depends
from pydantic import BaseModel
from typing import List, Optional
import json
import numpy as np
import redis
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.core.config import settings
from app.db.session import get_db

from app.services.wandb_service import wandb_service

# 로거 설정
logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------
# 1. 데이터 모델 정의
# ---------------------------------------------------------
class InferenceResult(BaseModel):
    job_id: int                 # post_id
    unified_vector: List[float] # 1152차원 벡터
    status: str                 # "completed" or "failed"

class FeedbackRequest(BaseModel):
    post_id: int
    correct_level: int

# ---------------------------------------------------------
# 2. 유틸리티: DB 및 Redis 연결 설정
# ---------------------------------------------------------
# Redis 연결
redis_kwargs = {"decode_responses": True}
if settings.REDIS_PASSWORD:
    # AWS ElastiCache는 보통 SSL(rediss://) 필요
    redis_url = f"rediss://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}"
    redis_kwargs["ssl_cert_reqs"] = None
else:
    redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}"

try:
    redis_client = redis.Redis.from_url(redis_url, **redis_kwargs)
    redis_client.ping()
    logger.info("Redis 연결 성공 (routes.py)")
except Exception as e:
    logger.error(f"Redis 연결 실패 (routes.py): {e}")

# MySQL 연결 (SQLAlchemy Core 사용 - 빠른 업데이트용)
db_url = f"mysql+pymysql://{settings.MYSQL_USER}:{settings.MYSQL_PASSWORD}@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DB}"
db_engine = create_engine(db_url, pool_recycle=3600)

# Qdrant 연결
qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

# ---------------------------------------------------------
# 3. API 엔드포인트: 결과 수신 (Modal Webhook)
# ---------------------------------------------------------
@router.post("/internal/inference-result")
async def receive_inference_result(
    result: InferenceResult,
    x_santa_token: Optional[str] = Header(None, alias="x-santa-token") # alias 중요!
):
    logger.info(f"[Webhook] 결과 수신 (Job ID: {result.job_id}, Status: {result.status})")

    # 1. 토큰 검증
    if x_santa_token != settings.SANTA_SECRET_TOKEN:
        logger.warning("승인되지 않은 접근 시도 (Token Mismatch)")
        raise HTTPException(status_code=403, detail="Unauthorized")

    if result.status != "completed" or not result.unified_vector:
        logger.warning("실패한 작업이므로 DB 업데이트를 건너뜁니다.")
        return {"status": "ignored"}

    try:
        # A. Qdrant에 벡터 저장
        # collection_name은 기존에 쓰시던 "santa_images"로 통일합니다.
        try:
            qdrant_client.upsert(
                collection_name="santa_images",
                points=[
                    models.PointStruct(
                        id=result.job_id,
                        vector=result.unified_vector,
                        payload={"post_id": result.job_id, "level": 0} # 초기엔 0, 아래에서 업데이트
                    )
                ]
            )
            logger.info(f"Qdrant 저장 완료 (ID: {result.job_id})")
        except Exception as q_err:
            logger.error(f"Qdrant 저장 실패: {q_err}")
            # Qdrant 실패해도 RDS 업데이트는 시도하도록 continue

        # B. 레벨 계산 (Centroid와 비교)
        centroids_data = redis_client.get("system:centroids")
        centroids = {}
        if centroids_data:
            centroids = json.loads(centroids_data)

        level = calculate_level(result.unified_vector)
        logger.info(f"📏 계산된 레벨: {level}")

        # C. MySQL 업데이트 (level, content_visible=1)
        with db_engine.connect() as conn:
            # 1. posts 테이블 업데이트
            stmt = text("""
                UPDATE posts 
                SET post_level = :lvl
                WHERE post_id = :pid
            """)
            conn.execute(stmt, {"lvl": level, "pid": result.job_id})
            
            conn.commit()
            
        logger.info(f"RDS 업데이트 완료 (Post ID: {result.job_id} -> Level {level})")

        # Qdrant Payload 업데이트 (레벨 확정)
        qdrant_client.set_payload(
            collection_name="santa_images",
            payload={"level": level},
            points=[result.job_id]
        )

        wandb_service.log_point(
            vector=result.unified_vector,
            point_type="post",
            point_id=str(result.job_id),
            level=level # 위에서 계산된 level
        )
        
    except Exception as e:
        logger.error(f"데이터 처리 중 에러: {e}")
        raise HTTPException(status_code=500, detail=str(e))


    return {"status": "success", "assigned_level": level}

# ---------------------------------------------------------
# 4. 레벨 계산 로직
# ---------------------------------------------------------
def calculate_level(target_vector: List[float]) -> int:
    try:
        data = redis_client.get("system:centroids")
        if not data:
            logger.warning("Redis에 Centroid 데이터가 없습니다! 기본값 5 반환")
            return 5
        
        centroids = json.loads(data)
        
        best_level = 5
        max_similarity = -1.0
        
        target_np = np.array(target_vector)
        target_norm = np.linalg.norm(target_np)

        if target_norm == 0: return 5

        for lvl_str, centroid_vec in centroids.items():
            c_np = np.array(centroid_vec)
            c_norm = np.linalg.norm(c_np)
            if c_norm == 0: continue
            
            similarity = np.dot(target_np, c_np) / (target_norm * c_norm)
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_level = int(lvl_str)
        
        return best_level

    except Exception as e:
        logger.error(f"레벨 계산 중 에러: {e}")
        return 5

# ---------------------------------------------------------
# 5. Qdrant 초기화 (유틸리티)
# ---------------------------------------------------------
@router.post("/setup/qdrant")
async def setup_qdrant():
    try:
        collection_name = "santa_images"
        collections = qdrant_client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        
        if exists:
            return {"message": f"Collection '{collection_name}' already exists."}

        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1152, 
                distance=models.Distance.COSINE
            ),
        )
        return {"message": f"Collection '{collection_name}' created successfully!"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))