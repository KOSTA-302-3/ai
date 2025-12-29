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
        """WandB Run이 없으면 초기화"""
        if wandb.run is None:
            try:
                wandb.init(
                    project=self.project_name,
                    job_type="production_monitoring",
                    resume="allow" 
                )
                self.initialized = True
            except Exception as e:
                logger.error(f"WandB 초기화 실패: {e}")

    def log_point(self, vector: list, point_type: str, point_id: str, level: int):
        """
        벡터 데이터를 WandB에 로깅합니다.
        - point_type: 'post' 또는 'centroid'
        - point_id: post_id 또는 level_id
        """
        try:
            self._ensure_init()
            
            table = wandb.Table(columns=["id", "type", "level", "embedding"])
            
            table.add_data(
                str(point_id),
                point_type,
                level,
                vector
            )
            
            wandb.log({"santa_vectors": table})
            
        except Exception as e:
            logger.error(f"WandB 로깅 실패: {e}")

wandb_service = WandBService()