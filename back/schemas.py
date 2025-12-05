from pydantic import BaseModel

# 1. 사용자가 보낼 데이터 형식 (Request)
class AIRequest(BaseModel):
    text: str
    
    # 예시 데이터를 문서에 보여주기 위한 설정
    class Config:
        json_schema_extra = {
            "example": {
                "text": "이 제품 정말 좋아요! 강력 추천합니다."
            }
        }

# 2. 서버가 응답할 데이터 형식 (Response)
class AIResponse(BaseModel):
    original_text: str
    result: str
    status: str