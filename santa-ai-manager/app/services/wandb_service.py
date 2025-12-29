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

    def log_point(self, vector: list, point_type: str, point_id: str, level: int):
        """단건 로깅 (routes.py용)"""
        try:
            self._ensure_init()
            if not vector: return

            table = wandb.Table(columns=["id", "type", "level", "embedding"])
            table.add_data(str(point_id), point_type, level, vector)
            wandb.log({"santa_vectors": table})
            
        except Exception as e:
            logger.error(f"WandB 로깅 실패: {e}")

    # 👇 [신규 추가] 여러 건을 한 번에 로깅하는 함수
    def log_batch(self, items: list):
        """
        items: [(vector, point_type, point_id, level), ...] 형태의 리스트
        """
        try:
            self._ensure_init()
            if not items: return

            table = wandb.Table(columns=["id", "type", "level", "embedding"])
            
            for item in items:
                # item unpacking: (vector, type, id, level) 순서 주의
                # 위 add_data 순서: id, type, level, vector
                vec, p_type, p_id, lvl = item
                table.add_data(str(p_id), p_type, lvl, vec)

            wandb.log({"santa_vectors": table})
            logger.info(f"WandB Batch 로깅 완료 ({len(items)}건)")

        except Exception as e:
            logger.error(f"WandB Batch 로깅 실패: {e}")

wandb_service = WandBService()