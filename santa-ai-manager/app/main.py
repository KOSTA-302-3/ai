from fastapi import FastAPI
import asyncio
from app.api.routes import router as api_router
from app.services.worker import start_worker

app = FastAPI(title="Project Santa AI Manager")

# API 경로 등록
app.include_router(api_router)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(start_worker())

@app.get("/health")
def health_check():
    return {"status": "ok"}
