# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # [프로젝트 기본 정보]
    PROJECT_NAME: str = "Santa-AI-Manager"
    VERSION: str = "1.0.0"
    ENV_MODE: str = "production"

    # [Redis 설정]
    REDIS_HOST: str
    REDIS_PORT: int = 6379
    REDIS_QUEUE_NAME: str = "queue:inference"

    # [Qdrant 설정]
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333

    # [보안 및 통신]
    CALLBACK_BASE_URL: str 
    SANTA_SECRET_TOKEN: str

    # [AWS S3 및 기타 키]
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str = "ap-southeast-2"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore", 
        case_sensitive = True
    )

# 싱글톤 객체 생성
settings = Settings()