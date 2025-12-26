import os
import sys
from sqlalchemy import create_engine, text
from qdrant_client import QdrantClient, models
from urllib.parse import quote_plus

# 프로젝트 설정 가져오기
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.core.config import settings

def init_system():
    print("\nQdrant 'santa_images' 컬렉션 생성 중...")
    try:
        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        collection_name = "santa_images"
        
        # 컬렉션 목록 조회
        collections = client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        
        if exists:
            print(f"Qdrant: '{collection_name}' 컬렉션이 이미 존재합니다.")
        else:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=1152,  # Modal 벡터 차원 수
                    distance=models.Distance.COSINE
                ),
            )
            print(f"Qdrant: '{collection_name}' 컬렉션 생성 완료!")

    except Exception as e:
        print(f"Qdrant 초기화 실패: {e}")

if __name__ == "__main__":
    init_system()