from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

# 1. 엔진 생성
engine = create_engine(
    settings.SQLALCHEMY_DATABASE_URI,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_size=10,
    max_overflow=20
)

# 2. 세션 공장
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 3. Base 클래스
Base = declarative_base()

# 4. 의존성 함수 (FastAPI용)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()