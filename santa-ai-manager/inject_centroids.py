import json
import redis
import os
import sys

# Qdrant 라이브러리 추가
from qdrant_client import QdrantClient
from qdrant_client.http import models

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.core.config import settings

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
    # 1. Redis 저장 (기존 코드 유지)
    # ---------------------------------------------------------
    try:
        # Spring 설정: spring.data.redis.ssl.enabled=true 
        # -> Python에서는 protocol='rediss' 및 ssl_cert_reqs=None 처리 필요
        
        # 비밀번호가 .env에 잘 들어갔는지 확인
        if not settings.REDIS_PASSWORD:
            print("오류: .env 파일에 'REDIS_PASSWORD'가 설정되지 않았습니다!")
            return

        redis_url = f"rediss://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}"
        
        print(f"Redis 보안 접속 시도 (SSL): {settings.REDIS_HOST}...")

        # ssl_cert_reqs=None: AWS 내부 통신 시 인증서 검증을 건너뛰어 에러 방지
        r = redis.Redis.from_url(
            redis_url, 
            decode_responses=True, 
            ssl_cert_reqs=None 
        )
        
        # 연결 테스트
        r.ping()
        print("Redis 연결 성공! (Auth & SSL OK)")

        # 데이터 저장
        redis_key = "system:centroids"
        r.set(redis_key, json.dumps(centroids_data))
        print(f"Redis Key '{redis_key}' 저장 완료!")
        
        # 검증
        if r.get(redis_key):
            print("Redis 데이터 주입 성공!")
        else:
            print("Redis 저장 실패 (데이터가 비어있음)")

    except Exception as e:
        print(f"Redis 처리 중 에러: {e}")
        return # Redis 실패 시 중단하려면 return 유지, 아니면 pass

    # ---------------------------------------------------------
    # 2. Qdrant 저장 (신규 추가됨)
    # ---------------------------------------------------------
    print("\nQdrant 데이터 주입 시작 (시각화용)...")
    try:
        # Qdrant 연결
        q_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        collection_name = "santa_centroids"
        vector_size = 1152  # SigLIP 모델 차원

        # 컬렉션 생성 (이미 있으면 다시 만듦)
        q_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )

        # 데이터 변환 (Dict -> PointStruct)
        points = []
        for level, vector in centroids_data.items():
            points.append(
                models.PointStruct(
                    id=int(level),  # ID는 레벨 숫자(0~5) 그대로 사용
                    vector=vector,
                    payload={
                        "level": int(level),
                        "type": "centroid",
                        "label": f"Level {level}"
                    }
                )
            )

        # 업로드
        q_client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"Qdrant 컬렉션 '{collection_name}'에 Centroid {len(points)}개 저장 완료!")

    except Exception as e:
        print(f"Qdrant 처리 중 에러: {e}")

    print("\n모든 작업 완료.")

if __name__ == "__main__":
    inject_centroids()