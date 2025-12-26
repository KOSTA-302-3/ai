from fastapi import APIRouter, Header, HTTPException, Depends
from pydantic import BaseModel
from typing import List
import logging

# [설정 및 서비스]
from app.core.config import settings
from app.services.centroid_service import CentroidService
from app.core.connections import redis_client

# [Qdrant - 벡터 DB]
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# [RDS - 데이터베이스]
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.post import Post

logger = logging.getLogger(__name__)
router = APIRouter()

# Centroid 서비스 인스턴스 생성
centroid_service = CentroidService()

# Qdrant 접속 정보
QDRANT_HOST = settings.QDRANT_HOST
QDRANT_PORT = settings.QDRANT_PORT

# === [ 요청/응답 모델 정의 ] ===

class InferenceResult(BaseModel):
    job_id: int        # Post ID (Integer)
    unified_vector: List[float]
    status: str

class FeedbackRequest(BaseModel):
    post_id: int
    correct_level: int

# === [ API 엔드포인트 ] ===
@router.get("/internal/test-redis")
async def test_redis():
    """Redis 연결 상태 확인용"""
    try:
        pong = await redis_client.ping()
        return {"redis_status": "connected" if pong else "failed"}
    except Exception as e:
        return {"redis_status": "error", "detail": str(e)}

@router.post("/inference-result")
async def receive_result(
    result: InferenceResult, 
    x_santa_token: str = Header(None),
    db: Session = Depends(get_db)
):
    """
    [Webhook] Modal GPU 서버로부터 추론 결과를 수신
    1. Centroid와 비교하여 레벨 결정 (코사인 유사도)
    2. Qdrant에 벡터 및 메타데이터(레벨 포함) 저장
    3. RDS(MySQL) Post 테이블의 level 업데이트
    """
    # 1. 보안 토큰 검증
    if x_santa_token != settings.SANTA_SECRET_TOKEN:
        logger.warning(f"승인되지 않은 접근 시도! Post ID: {result.job_id}")
        raise HTTPException(status_code=403, detail="Unauthorized")

    logger.info(f"[Webhook] 결과 수신: Post ID {result.job_id}")
    
    try:
        # 2. Centroid 로드 및 레벨 결정
        centroids = await centroid_service.get_centroids()
        
        if centroids:
            determined_level = centroid_service.determine_level(result.unified_vector, centroids)
            logger.info(f"AI 판단 레벨: {determined_level} (Post ID: {result.job_id})")
        else:
            logger.warning("Centroid가 초기화되지 않았습니다. 기본값 0을 사용합니다.")
            determined_level = 0

        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        client.upsert(
            collection_name="santa_images",
            points=[
                PointStruct(
                    id=result.job_id,  
                    vector=result.unified_vector,
                    payload={
                        "post_id": result.job_id,
                        "status": "completed",
                        "level": determined_level
                    }
                )
            ]
        )
        logger.info(f"Qdrant 저장 완료 (ID: {result.job_id})")

        target_post = db.query(Post).filter(Post.post_id == result.job_id).first()
        
        if target_post:
            target_post.level = determined_level
            db.commit()
            logger.info(f"✅ RDS Level 업데이트 완료 (Post ID: {result.job_id} -> Level {determined_level})")
        else:
            logger.warning(f"⚠️ RDS에서 해당 게시물을 찾을 수 없음 (ID: {result.job_id})")

    except Exception as e:
        logger.error(f"처리 중 에러 발생: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
    return {"status": "success"}

@router.post("/feedback")
async def process_feedback(
    feedback: FeedbackRequest,
    db: Session = Depends(get_db)
):
    """
    [Feedback Loop] 사용자 피드백 반영 및 모델 미세 조정
    1. 해당 게시물의 벡터와 현재 레벨을 조회
    2. 레벨이 다르면 Centroid 위치 조정 (Pull, Push, Repulsion)
    3. DB 및 Qdrant의 해당 게시물 레벨 즉시 수정
    """
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        points = client.retrieve(
            collection_name="santa_images",
            ids=[feedback.post_id],
            with_vectors=True
        )
        
        if not points:
            raise HTTPException(status_code=404, detail="Post not found in Vector DB")
            
        post_vector = points[0].vector
        current_level = points[0].payload.get("level", 0)

        if current_level != feedback.correct_level:
            logger.info(f"[Feedback] Post {feedback.post_id}: Level {current_level} -> {feedback.correct_level}")
            
            await centroid_service.adjust_centroid(
                vector=post_vector,
                old_level=current_level,
                correct_level=feedback.correct_level
            )
            
            points[0].payload["level"] = feedback.correct_level
            client.upsert(collection_name="santa_images", points=points)
            
            post = db.query(Post).filter(Post.post_id == feedback.post_id).first()
            if post:
                post.level = feedback.correct_level
                db.commit()
            
            return {"status": "feedback_applied", "message": "Centroids adjusted and post updated."}
        
        return {"status": "no_change", "message": "Levels match, no adjustment needed."}

    except Exception as e:
        logger.error(f"피드백 처리 중 에러: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/setup/qdrant")
async def setup_qdrant():
    """Qdrant 컬렉션(방) 생성 유틸리티"""
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collection_name = "santa_images"
        
        collections = client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        
        if exists:
            return {"message": f"Collection '{collection_name}' already exists."}

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1152, 
                distance=Distance.COSINE
            ),
        )
        return {"message": f"Collection '{collection_name}' created successfully!"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))