from sqlalchemy import Column, Integer
from app.db.session import Base

class Post(Base):
    __tablename__ = "posts" # 실제 RDS 테이블 이름

    post_id = Column(Integer, primary_key=True, index=True)
    level = Column("post_level", Integer, default=1)