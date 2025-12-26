from fastapi import FastAPI
import asyncio
from app.api.routes import router as api_router
from app.services.worker import start_worker
from app.db.init_db import init_system
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_system() 
        print("시스템 초기화 완료 (DB/Qdrant)")
    except Exception as e:
        print(f"시스템 초기화 실패: {e}")

    # 2. 백그라운드 워커 실행
    # (워커를 변수에 담아두면 나중에 제어하기 좋습니다, 일단 실행만 함)
    asyncio.create_task(start_worker())
    print("백그라운드 워커 시작됨")
    
    yield
    
    print("서버 종료 중...")

app = FastAPI(title="Project Santa AI Manager", lifespan=lifespan)

app.include_router(api_router)

@app.get("/health")
def health_check():
    return {"status": "ok"}