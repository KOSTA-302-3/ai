# push_job.py (수정본)
import redis
import json
import os
from app.core.config import settings

def push_test_job():
    r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)

    test_job = {
        "job_id": "santa-refactor-test",
        "image_urls": ["https://images.unsplash.com/photo-1543508282-6319a3e2621f"],
        "content": "하드코딩 제거 테스트 중!",
        "post_id": "post-123"
    }

    # 공유된 변수로 데이터 전송
    r.lpush(settings.REDIS_QUEUE_NAME, json.dumps(test_job))
    print(f"[{settings.REDIS_QUEUE_NAME}] 에 일감을 던졌습니다!")

if __name__ == "__main__":
    push_test_job()