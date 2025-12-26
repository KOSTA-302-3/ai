import json
import numpy as np
import logging
from app.core.connections import redis_client

logger = logging.getLogger(__name__)

class CentroidService:
    REDIS_KEY = "system:centroids"
    
    LEARNING_RATE = 0.02   
    REPULSION_RATE = 0.01  
    SIMILARITY_THRESHOLD = 0.95

    async def get_centroids(self):
        data = await redis_client.get(self.REDIS_KEY)
        if not data:
            return None
        return json.loads(data)

    async def save_centroids(self, centroids: dict):
        await redis_client.set(self.REDIS_KEY, json.dumps(centroids))

    def _normalize(self, vector):
        np_vec = np.array(vector)
        norm = np.linalg.norm(np_vec)
        if norm == 0:
            return np_vec
        return np_vec / norm

    def determine_level(self, vector: list, centroids: dict) -> int:
        vec_np = self._normalize(vector)
        
        max_sim = -2.0 
        best_level = 0

        for level_str, cent_vec in centroids.items():
            cent_np = np.array(cent_vec)
            
            similarity = np.dot(vec_np, cent_np)
            
            if similarity > max_sim:
                max_sim = similarity
                best_level = int(level_str)
        
        return best_level

    async def adjust_centroid(self, vector: list, old_level: int, correct_level: int):
        centroids = await self.get_centroids()
        if not centroids:
            return

        vec_input = self._normalize(vector)

        target_key = str(correct_level)
        if target_key in centroids:
            target_vec = np.array(centroids[target_key])
            target_vec = target_vec + self.LEARNING_RATE * (vec_input - target_vec)
            centroids[target_key] = self._normalize(target_vec).tolist()

        old_key = str(old_level)
        if old_key in centroids and old_level != correct_level:
            origin_vec = np.array(centroids[old_key])
            origin_vec = origin_vec - self.LEARNING_RATE * (vec_input - origin_vec)
            centroids[old_key] = self._normalize(origin_vec).tolist()

        await self._apply_repulsion(centroids, target_key)
        
        await self.save_centroids(centroids)
        return centroids

    async def _apply_repulsion(self, centroids, target_key):
        target_vec = np.array(centroids[target_key])

        for lvl, vec in centroids.items():
            if lvl == target_key:
                continue
            
            neighbor_vec = np.array(vec)
            similarity = np.dot(target_vec, neighbor_vec)

            if similarity > self.SIMILARITY_THRESHOLD:
                push_dir = neighbor_vec - target_vec
                
                if np.linalg.norm(push_dir) == 0:
                    push_dir = np.random.rand(len(target_vec)) - 0.5

                neighbor_vec = neighbor_vec + (self.REPULSION_RATE * push_dir)
                
                centroids[lvl] = self._normalize(neighbor_vec).tolist()
                logger.info(f"반발력 작용: {target_key}번이 {lvl}번을 밀어냈습니다. (유사도: {similarity:.4f})")