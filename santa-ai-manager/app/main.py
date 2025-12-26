from fastapi import FastAPI
import asyncio
from app.api.routes import router as api_router
from app.services.worker import start_worker
from app.db.init_db import init_system
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # [시작 시 실행]
    # 여기에 초기화 함수를 넣어서 서버 켜질 때 DB/Qdrant 세팅을 보장합니다.
    try:
        init_system() 
        print("✅ 시스템 초기화 완료")
    except Exception as e:
        print(f"❌ 시스템 초기화 실패: {e}")
    
    yield

app = FastAPI(title="Project Santa AI Manager")

# API 경로 등록
app.include_router(api_router)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(start_worker())

@app.get("/health")
def health_check():
    return {"status": "ok"}
