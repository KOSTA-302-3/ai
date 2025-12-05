from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from model import FakeAIModel
from schemas import AIRequest, AIResponse

# 1. 전역 변수로 모델 인스턴스 생성 (아직 로드 안 됨)
ai_model = FakeAIModel()

# 2. Lifespan: 서버가 켜지고 꺼질 때 실행할 작업 정의
@asynccontextmanager
async def lifespan(app: FastAPI):
    # [시작 시 실행]
    ai_model.load_model()  # 모델 로딩 (약 2초 소요)
    yield
    # [종료 시 실행]
    print("서버 종료: 리소스를 정리합니다.")

# 3. FastAPI 앱 생성 (lifespan 적용)
app = FastAPI(lifespan=lifespan)

# 기본 경로 테스트
@app.get("/")
def read_root():
    return {"message": "AI Server is running!"}

# 4. AI 추론 엔드포인트 생성
@app.post("/predict", response_model=AIResponse)
async def predict_sentiment(request: AIRequest):
    try:
        # 사용자가 보낸 텍스트 가져오기
        input_text = request.text
        
        # 모델에 추론 요청
        prediction = ai_model.predict(input_text)
        
        # 결과 반환 (schemas.py의 AIResponse 형식에 맞게)
        return AIResponse(
            original_text=input_text,
            result=prediction,
            status="success"
        )
    except Exception as e:
        # 에러 발생 시 500 에러 반환
        raise HTTPException(status_code=500, detail=str(e))