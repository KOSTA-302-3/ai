import modal
import os

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch", "torchvision", "transformers", "pillow", 
        "boto3", "accelerate", "sentencepiece", "protobuf", "timm"
    )
)

app = modal.App("santa", image=image)
model_volume = modal.Volume.from_name("santa-models", create_if_missing=True)
MODEL_PATH = "/models/siglip_best.pth"

@app.function(
    gpu="T4", 
    volumes={"/models": model_volume}, 
    secrets=[modal.Secret.from_name("santa-aws-secret")],
    timeout=900 
)
def run_inference(image_urls: list, content: str, job_id: str, callback_url: str, secret_token: str):
    import torch
    import requests
    from PIL import Image
    from io import BytesIO
    from transformers import AutoModel, AutoProcessor
    from botocore.config import Config
    import numpy as np

    # 모델 가중치 로드
    if not os.path.exists(MODEL_PATH):
        s3 = boto3.client("s3", config=Config(region_name='ap-southeast-2'))
        s3.download_file("kosta-santa-s3", "siglip_best.pth", MODEL_PATH)
        model_volume.commit()

    # SigLIP 모델 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "google/siglip-so400m-patch14-384" 
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model.eval()

    all_vectors = []

    # 이미지 벡터 추출
    for url in image_urls:
        try:
            res = requests.get(url, timeout=10)
            img = Image.open(BytesIO(res.content)).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                v = model.get_image_features(**inputs).cpu().numpy()[0]
                all_vectors.append(v)
        except: continue

    # 텍스트 벡터 추출
    if content:
        text_inputs = processor(text=[content], padding="max_length", truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            v = model.get_text_features(**text_inputs).cpu().numpy()[0]
            all_vectors.append(v)

    # 통합 및 단위벡터화
    unified_vector = None
    if all_vectors:
        # 모든 벡터 평균
        combined = np.mean(all_vectors, axis=0)
        
        # L2 정규화: 벡터의 길이를 1로 변환
        norm = np.linalg.norm(combined)
        if norm > 0:
            unified_vector = (combined / norm).tolist() # 단위벡터화 완료
        else:
            unified_vector = combined.tolist()

    # Webhook 전송
    payload = {"job_id": job_id, "unified_vector": unified_vector, "status": "completed"}
    requests.post(callback_url, json=payload, headers={"x-santa-token": secret_token})
    
    return {"status": "success"}