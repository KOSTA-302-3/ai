import json
import numpy as np
import logging
import asyncio
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http import models

from app.core.config import settings
from app.core.connections import redis_client
from app.db.session import SessionLocal
from app.models.post import Post

from app.services.wandb_service import wandb_service

logger = logging.getLogger(__name__)

class CentroidService:
    REDIS_KEY = "system:centroids"
    
    # 하이퍼파라미터 (기존 설정 유지)
    LEARNING_RATE = 0.02   
    REPULSION_RATE = 0.01  
    SIMILARITY_THRESHOLD = 0.95

    def __init__(self):
        # Qdrant 클라이언트는 동기 방식으로 사용 (데이터 처리를 위해)
        self.qdrant = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

    async def get_centroids(self) -> Dict[str, List[float]]:
        """Redis에서 Centroid 정보를 가져옵니다."""
        data = await redis_client.get(self.REDIS_KEY)
        if not data:
            logger.warning("Redis에 Centroid 데이터가 없습니다.")
            return {}
        return json.loads(data)

    async def save_centroids(self, centroids: Dict[str, List[float]]):
        """Redis와 Qdrant에 변경된 Centroid 정보를 저장합니다."""
        
        # 1. Redis 저장
        await redis_client.set(self.REDIS_KEY, json.dumps(centroids))
        
        # 2. Qdrant 저장 (시각화용)
        try:
            qdrant_points = []
            wandb_items = []  # WandB용 리스트

            for level, vector in centroids.items():
                # Qdrant용 데이터 준비
                qdrant_points.append(
                    models.PointStruct(
                        id=int(level), 
                        vector=vector,
                        payload={"level": int(level), "type": "centroid", "updated_at": "now"}
                    )
                )
                
                # [수정됨] WandB용 데이터 수집 (보내지 않고 리스트에 담기만 함)
                wandb_items.append((
                    vector,                 # vector
                    "centroid",             # type
                    f"centroid_lv{level}",  # id
                    int(level)              # level
                ))
            
            # Qdrant 실행
            self.qdrant.upsert(collection_name="santa_centroids", points=qdrant_points)
            
            # [수정됨] WandB 일괄 전송!
            wandb_service.log_batch(wandb_items)
            
            logger.info("Redis, Qdrant, WandB 업데이트 완료")
            
        except Exception as e:
            logger.error(f"저장 중 에러 발생: {e}")

    def _normalize(self, vector: List[float]) -> np.ndarray:
        """벡터 정규화 (L2 Norm)"""
        np_vec = np.array(vector)
        norm = np.linalg.norm(np_vec)
        if norm == 0:
            return np_vec
        return np_vec / norm

    def determine_level(self, vector: List[float], centroids: Dict[str, List[float]]) -> int:
        """
        벡터와 Centroid 간의 코사인 유사도를 계산하여 가장 가까운 레벨을 반환합니다.
        """
        vec_np = self._normalize(vector)
        
        max_sim = -2.0 
        best_level = 5 # 기본값

        for level_str, cent_vec in centroids.items():
            cent_np = np.array(cent_vec)
            # 코사인 유사도: dot product (이미 정규화되어 있다고 가정)
            similarity = np.dot(vec_np, cent_np)
            
            if similarity > max_sim:
                max_sim = similarity
                best_level = int(level_str)
        
        return best_level

    async def process_feedback_job(self, feedback_data: dict):
        post_id = feedback_data.get("job_id")
        correct_level = feedback_data.get("level")

        if not post_id or not correct_level:
            logger.error("잘못된 피드백 데이터입니다.")
            return

        # 1. 해당 Post의 벡터 가져오기 (Qdrant)
        vector = self._fetch_vector_from_qdrant(post_id)
        if vector is None:
            logger.error(f"Post {post_id}의 벡터를 찾을 수 없어 피드백을 건너뜁니다.")
            return

        # 2. 현재 Centroid 가져오기
        centroids = await self.get_centroids()
        if not centroids:
            logger.error("초기 Centroids가 없습니다.")
            return

        # 3. Centroid 조정 (학습 로직 적용)
        # 현재 레벨 계산 (비교용)
        current_calculated_level = self.determine_level(vector, centroids)
        
        updated_centroids = await self._adjust_centroids_logic(
            centroids, vector, correct_level, current_calculated_level
        )

        # 4. Redis에 업데이트
        await self.save_centroids(updated_centroids)

        # 5. RDS의 모든 Post Level 재계산 및 업데이트 (Heavy Task)
        # 동기 작업이므로 비동기 루프를 차단하지 않도록 run_in_executor 사용 권장
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._recalculate_all_posts_levels, updated_centroids)

    # ---------------------------------------------------------
    # [내부 로직] - 학습 및 데이터 처리
    # ---------------------------------------------------------
    async def _adjust_centroids_logic(self, centroids, vector, correct_level, old_level):
        vec_input = self._normalize(vector)
        target_key = str(correct_level)
        old_key = str(old_level)

        # 1. 정답 레벨 당기기 (Attraction)
        if target_key in centroids:
            target_vec = np.array(centroids[target_key])
            # 공식: New = Old + LR * (Input - Old)
            target_vec = target_vec + self.LEARNING_RATE * (vec_input - target_vec)
            centroids[target_key] = self._normalize(target_vec).tolist()

        # 2. 기존(오답) 레벨 밀어내기 (Unlearning)
        # 사용자가 지정한 레벨과 기계가 예측한 레벨이 다를 경우에만 수행
        if old_key in centroids and old_level != correct_level:
            origin_vec = np.array(centroids[old_key])
            # 공식: New = Old - LR * (Input - Old)
            origin_vec = origin_vec - self.LEARNING_RATE * (vec_input - origin_vec)
            centroids[old_key] = self._normalize(origin_vec).tolist()

        # 3. 군집 간 반발력 적용 (Cluster Repulsion)
        # 정답 레벨이 이동함에 따라 너무 가까워진 다른 레벨들을 밀어냄
        centroids = self._apply_repulsion(centroids, target_key)
        
        return centroids

    def _apply_repulsion(self, centroids, target_key):
        if target_key not in centroids:
            return centroids

        target_vec = np.array(centroids[target_key])

        for lvl, vec in centroids.items():
            if lvl == target_key:
                continue
            
            neighbor_vec = np.array(vec)
            similarity = np.dot(target_vec, neighbor_vec)

            # 유사도가 임계값보다 높으면 서로 밀어냄
            if similarity > self.SIMILARITY_THRESHOLD:
                # 밀어낼 방향 벡터
                push_dir = neighbor_vec - target_vec
                
                # 완전히 겹칠 경우 랜덤 방향으로
                if np.linalg.norm(push_dir) == 0:
                    push_dir = np.random.rand(len(target_vec)) - 0.5

                # Repulsion 적용
                neighbor_vec = neighbor_vec + (self.REPULSION_RATE * push_dir)
                centroids[lvl] = self._normalize(neighbor_vec).tolist()
                
        return centroids

    def _fetch_vector_from_qdrant(self, post_id: int) -> Optional[List[float]]:
        try:
            points = self.qdrant.retrieve(
                collection_name="santa_images",
                ids=[post_id],
                with_vectors=True
            )
            if points:
                return points[0].vector
            return None
        except Exception as e:
            logger.error(f"Qdrant 벡터 조회 실패 (ID: {post_id}): {e}")
            return None

    def _recalculate_all_posts_levels(self, centroids: dict):
        """
        [Heavy Task - Optimized] 
        Scroll API를 사용하여 배치 단위로 처리 (성능 최적화)
        """
        logger.info("전체 RDS Post Level 재계산 시작 (Batch Processing)...")
        db: Session = SessionLocal()
        total_updates = 0
        processed_count = 0
        
        # Scroll 커서 초기화
        next_offset = None
        batch_size = 100  # 한 번에 가져올 데이터 양

        try:
            while True:
                # 1. Qdrant에서 배치 단위로 벡터 가져오기
                points, next_offset = self.qdrant.scroll(
                    collection_name="santa_images",
                    limit=batch_size,
                    offset=next_offset,
                    with_vectors=True, # 벡터 필수
                    with_payload=False # Payload는 불필요
                )

                if not points:
                    break

                # 2. 가져온 배치 데이터에 대해 레벨 계산 및 DB 업데이트
                for point in points:
                    try:
                        post_id = int(point.id) # Qdrant ID = Post ID
                        new_level = self.determine_level(point.vector, centroids)

                        # RDS 업데이트
                        # (주의: 여기서도 매번 SELECT/UPDATE 하면 느릴 수 있으나, 
                        #  SQLAlchemy 캐싱 덕분에 1건씩 처리보다 훨씬 빠릅니다.
                        #  더 최적화하려면 bulk_update_mappings를 써야 합니다.)
                        post = db.query(Post).filter(Post.post_id == post_id).first()
                        
                        if post and post.level != new_level:
                            post.level = new_level
                            total_updates += 1
                            
                    except ValueError:
                        continue # ID가 숫자가 아닌 경우 등 예외 처리

                # 3. 배치 단위 커밋 (트랜잭션 부하 조절)
                db.commit()
                processed_count += len(points)
                
                # 로그 (진행 상황)
                if processed_count % 1000 == 0:
                    logger.info(f"재계산 진행 중: {processed_count}개 처리 완료...")

                # 다음 페이지가 없으면 루프 종료
                if next_offset is None:
                    break

            logger.info(f"재계산 완료. 총 {total_updates}개의 게시물 레벨이 변경되었습니다.")

        except Exception as e:
            db.rollback()
            logger.error(f"RDS 업데이트 중 오류: {e}")
        finally:
            db.close()