import json
import redis
import os
import sys

# Qdrant & WandB 라이브러리
from qdrant_client import QdrantClient
from qdrant_client.http import models
import wandb  # 스크립트 종료 처리를 위해 직접 import

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.core.config import settings
# 방금 만든 wandb_service 가져오기
from app.services.wandb_service import wandb_service

def inject_centroids():
    print("Centroid 데이터 주입 시작...")

    json_file = "initial_centroids.json"
    if not os.path.exists(json_file):
        print(f"오류: '{json_file}' 파일이 없습니다.")
        return

    with open(json_file, "r", encoding="utf-8") as f:
        centroids_data = json.load(f)

    print(f"JSON 로드 완료 (총 {len(centroids_data)}개 레벨)")

    # ---------------------------------------------------------
    # 1. Redis 저장
    # ---------------------------------------------------------
    try:
        if not settings.REDIS_PASSWORD:
            print("오류: .env 파일에 'REDIS_PASSWORD'가 설정되지 않았습니다!")
            return

        redis_url = f"rediss://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}"
        
        # ssl_cert_reqs=None 추가 (AWS ElastiCache 등 호환성)
        r = redis.Redis.from_url(
            redis_url, 
            decode_responses=True, 
            ssl_cert_reqs=None 
        )
        
        r.ping()
        print("Redis 연결 성공!")

        redis_key = "system:centroids"
        r.set(redis_key, json.dumps(centroids_data))
        print(f"Redis Key '{redis_key}' 저장 완료!")

    except Exception as e:
        print(f"Redis 처리 중 에러: {e}")
        return

    # ---------------------------------------------------------
    # 2. Qdrant 저장 (시각화용 컬렉션)
    # ---------------------------------------------------------
    print("\nQdrant 데이터 주입 시작...")
    try:
        q_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        
        # A. santa_centroids (순수 Centroid 저장소)
        collection_name = "santa_centroids"
        
        if q_client.collection_exists(collection_name):
            q_client.delete_collection(collection_name)
            
        q_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1152,
                distance=models.Distance.COSINE
            )
        )

        points = []
        for level, vector in centroids_data.items():
            points.append(
                models.PointStruct(
                    id=int(level),
                    vector=vector,
                    payload={"level": int(level), "type": "centroid", "label": f"Level {level}"}
                )
            )

        q_client.upsert(collection_name=collection_name, points=points)
        print(f"Qdrant 컬렉션 '{collection_name}' 저장 완료!")

        # B. santa_images (시각화 꼼수용 병합)
        if q_client.collection_exists("santa_images"):
            image_points = []
            for level, vector in centroids_data.items():
                image_points.append(
                    models.PointStruct(
                        id=100000000 + int(level), # ID 충돌 방지
                        vector=vector,
                        payload={"level": int(level), "type": "centroid", "label": f"CENTER_LV_{level}"}
                    )
                )
            q_client.upsert(collection_name="santa_images", points=image_points)
            print(f"Qdrant 컬렉션 'santa_images'에 병합 완료!")

    except Exception as e:
        print(f"Qdrant 처리 중 에러: {e}")

    # ---------------------------------------------------------
    # 3. [NEW] WandB 초기값 로깅
    # ---------------------------------------------------------
    print("\nWandB에 초기 Centroid 일괄 기록 중...")
    try:
        # 1. 빈 테이블 생성 (컬럼 정의)
        table = wandb.Table(columns=["id", "type", "level", "embedding"])

        # 2. 루프 돌면서 데이터 '추가'만 함 (전송 X)
        for level, vector in centroids_data.items():
            table.add_data(
                f"init_centroid_lv{level}", # id
                "centroid",                 # type
                int(level),                 # level
                vector                      # vector
            )

        # 3. 데이터가 꽉 찬 테이블을 "한 번만" 전송
        # 이렇게 해야 Step 0에 모든 점이 같이 찍힙니다.
        wandb.log({"santa_vectors": table})
        
        # 전송 완료 대기
        wandb.finish()
        print(" WandB 로깅 완료! (모든 레벨이 한 화면에 보입니다)")
        
    except Exception as e:
        print(f"WandB 로깅 실패: {e}")

    print("\n모든 작업 완료.")

if __name__ == "__main__":
    inject_centroids()