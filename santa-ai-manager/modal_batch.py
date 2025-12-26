import modal
import sys
import os

batch_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "torch", 
        "torchvision", 
        "transformers", 
        "pillow", 
        "pymysql", 
        "sqlalchemy", 
        "redis", 
        "qdrant-client", 
        "scikit-learn",
        "requests",
        "numpy",
        "accelerate",
        "sentencepiece"
    )
)

app = modal.App("santa-batch", image=batch_image)

model_volume = modal.Volume.from_name("santa-models", create_if_missing=True)
secrets = [modal.Secret.from_name("santa-aws-secret")]

MODEL_PATH = "/models/siglip_best.pth"

@app.function(
    gpu="T4",
    volumes={"/models": model_volume},
    secrets=secrets,
    timeout=3600
)
def run_batch_recalculation():
    """
    Centroid 재계산
    - posts 테이블과 image_sources 테이블을 JOIN하여 데이터 조회
    - post_level (FLOAT) -> int로 변환하여 사용
    - SigLIP으로 멀티 모달 벡터 생성 -> 통합 벡터 -> Centroid 갱신
    """
    import torch
    import numpy as np
    import requests
    import json
    import redis
    import pymysql
    from PIL import Image
    from io import BytesIO
    from sqlalchemy import create_engine, text
    from transformers import AutoModel, AutoProcessor
    from qdrant_client import QdrantClient, models

    print("[Batch] 멀티모달 Centroid 재계산 작업 시작 (Schema Sync)")

    # ---------------------------------------------------------
    # 1. DB 및 Redis 연결
    # ---------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    db_url = f"mysql+pymysql://{os.environ['MYSQL_USER']}:{os.environ['MYSQL_PASSWORD']}@{os.environ['MYSQL_HOST']}:{os.environ['MYSQL_PORT']}/{os.environ['MYSQL_DB']}"
    engine = create_engine(db_url)

    r = redis.Redis(
        host=os.environ['REDIS_HOST'], 
        port=int(os.environ['REDIS_PORT']), 
        decode_responses=True
    )

    # ---------------------------------------------------------
    # 2. 모델 로드
    # ---------------------------------------------------------
    print("🧠 SigLIP 모델 로딩 중...")
    model_name = "google/siglip-so400m-patch14-384"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    if os.path.exists(MODEL_PATH):
        print(f"📂 학습된 가중치 로드: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    
    model.eval()

    # ---------------------------------------------------------
    # 3. 데이터 로드 (Posts + Image Sources JOIN)
    # ---------------------------------------------------------
    print("📥 RDS 데이터 조회 중...")
    
    query_str = """
        SELECT 
            p.post_id,
            p.content,
            CAST(p.post_level AS UNSIGNED) as level,
            JSON_ARRAYAGG(i.source) as image_urls
        FROM posts p
        LEFT JOIN image_sources i ON p.post_id = i.post_id
        WHERE p.post_level BETWEEN 1 AND 10
        GROUP BY p.post_id, p.content, p.post_level
    """

    with engine.connect() as conn:
        posts = conn.execute(text(query_str)).fetchall()

    print(f"📊 처리 대상 게시물: {len(posts)}개")
    
    level_vectors_map = {i: [] for i in range(1, 11)}
    
    success_cnt = 0
    fail_cnt = 0

    # ---------------------------------------------------------
    # 4. 루프: 게시물별 통합 벡터 생성
    # ---------------------------------------------------------
    for row in posts:
        pid, content, level, img_urls_json = row
        
        # level이 float->int 변환 과정에서 범위 벗어날 수 있으므로 안전장치
        if not (1 <= level <= 10):
            continue

        temp_vectors = []

        try:
            # A. 이미지 벡터화 (JSON 문자열 파싱)
            if img_urls_json:
                try:
                    # MySQL JSON_ARRAYAGG 결과가 문자열로 넘어오면 파싱
                    url_list = json.loads(img_urls_json) if isinstance(img_urls_json, str) else img_urls_json
                    
                    # null 값이 리스트에 섞일 수 있으므로 필터링
                    url_list = [u for u in url_list if u]

                    for url in url_list:
                        try:
                            res = requests.get(url, timeout=5)
                            if res.status_code == 200:
                                img = Image.open(BytesIO(res.content)).convert("RGB")
                                inputs = processor(images=img, return_tensors="pt").to(device)
                                with torch.no_grad():
                                    v = model.get_image_features(**inputs).cpu().numpy()[0]
                                    temp_vectors.append(v)
                        except Exception:
                            continue 
                except Exception as e:
                    print(f"⚠️ 이미지 처리 실패 (ID: {pid}): {e}")

            # B. 텍스트 벡터화
            if content and isinstance(content, str) and len(content.strip()) > 0:
                text_inputs = processor(text=[content], padding="max_length", truncation=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    v_text = model.get_text_features(**text_inputs).cpu().numpy()[0]
                    temp_vectors.append(v_text)

            # C. 통합 (Mean & Normalize)
            if temp_vectors:
                combined_vec = np.mean(temp_vectors, axis=0)
                
                norm = np.linalg.norm(combined_vec)
                if norm > 0:
                    final_vector = combined_vec / norm
                else:
                    final_vector = combined_vec

                level_vectors_map[level].append(final_vector)
                success_cnt += 1
            else:
                fail_cnt += 1

        except Exception as e:
            print(f"치명적 에러 (ID: {pid}): {e}")
            fail_cnt += 1
            
        if (success_cnt + fail_cnt) % 50 == 0:
            print(f"진행률: {success_cnt + fail_cnt}/{len(posts)}")

    # ---------------------------------------------------------
    # 5. Centroid 계산 및 저장
    # ---------------------------------------------------------
    print("Centroid 산출 중...")
    new_centroids = {}

    for lvl in range(1, 11):
        vecs = np.array(level_vectors_map[lvl])
        
        if len(vecs) > 0:
            mean_v = np.mean(vecs, axis=0)
            norm_v = mean_v / np.linalg.norm(mean_v)
            new_centroids[str(lvl)] = norm_v.tolist()
            print(f"  - Level {lvl}: {len(vecs)}개 게시물 사용")
        else:
            print(f"Level {lvl}: 데이터 부족으로 갱신 스킵")

    if new_centroids:
        r.set("system:centroids", json.dumps(new_centroids))
        print(f"Centroid 업데이트 완료! (총 {len(new_centroids)}개 레벨)")
    else:
        print("갱신된 Centroid가 없습니다.")

    return {"status": "success", "updated_levels": list(new_centroids.keys())}

if __name__ == "__main__":
    with app.run():
        run_batch_recalculation.remote()