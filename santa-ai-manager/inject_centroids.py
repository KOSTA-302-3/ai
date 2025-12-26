import json
import redis
import os
import sys

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
    # SSL + Password 접속 설정
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

    except Exception as e:
        print(f"Redis 연결 실패: {e}")
        return

    # 3. 데이터 저장
    redis_key = "system:centroids"
    try:
        r.set(redis_key, json.dumps(centroids_data))
        print(f"Redis Key '{redis_key}' 저장 완료!")
        
        # 검증
        if r.get(redis_key):
            print("데이터 주입 성공! 모든 준비 완료.")
        else:
            print("저장 실패 (데이터가 비어있음)")

    except Exception as e:
        print(f"데이터 저장 중 에러: {e}")

if __name__ == "__main__":
    inject_centroids()