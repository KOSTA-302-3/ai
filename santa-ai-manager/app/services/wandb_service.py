import wandb
import os
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class WandBService:
    def __init__(self):
        self.project_name = os.getenv("WANDB_PROJECT", "santa-ai-manager")
        self.initialized = False

    def _ensure_init(self):
        if wandb.run is None:
            try:
                if hasattr(settings, "WANDB_API_KEY") and settings.WANDB_API_KEY:
                    wandb.login(key=settings.WANDB_API_KEY)
                
                wandb.init(
                    project=self.project_name,
                    job_type="production_monitoring",
                    resume="allow"
                )
                self.initialized = True
            except Exception as e:
                logger.error(f"WandB 초기화 실패: {e}")

    def log_batch(self, items: list):
        """Centroid 업데이트용 (기존 유지)"""
        try:
            self._ensure_init()
            if not items: return

            table = wandb.Table(columns=["id", "type", "level", "embedding"])
            for item in items:
                vec, p_type, p_id, lvl = item
                table.add_data(str(p_id), p_type, lvl, vec)

            wandb.log({"santa_vectors": table})
            logger.info(f"WandB Batch 로깅 완료 ({len(items)}건)")

        except Exception as e:
            logger.error(f"WandB Batch 로깅 실패: {e}")

    # 👇 [신규] Post 1개와 현재 Centroid들을 묶어서 로깅
    def log_inference(self, post_vector: list, post_id: str, post_level: int, centroids: dict):
        try:
            self._ensure_init()
            
            # 테이블 컬럼 정의
            table = wandb.Table(columns=["id", "type", "level", "embedding"])

            # 1. 주인공 (Post) 추가
            table.add_data(
                str(post_id), 
                "post", 
                post_level, 
                post_vector
            )

            # 2. 조연 (Current Centroids) 함께 추가
            # 이걸 같이 넣어줘야 화면에서 비교가 됩니다.
            if centroids:
                for level, vector in centroids.items():
                    table.add_data(
                        f"curr_centroid_lv{level}", # ID로 현재 상태임을 표시
                        "current_centroid",         # Type을 다르게 주어 모양 구분 가능
                        int(level),
                        vector
                    )

            # 전송
            wandb.log({"santa_vectors": table})
            
        except Exception as e:
            logger.error(f"WandB Inference 로깅 실패: {e}")

wandb_service = WandBService()