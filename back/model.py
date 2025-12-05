import time

class FakeAIModel:
    def __init__(self):
        self.is_loaded = False

    def load_model(self):
        """
        AI 모델을 메모리에 올리는 과정을 시뮬레이션 합니다.
        실제로는 여기서 torch.load() 같은 무거운 작업이 일어납니다.
        """
        print("모델 로딩 시작... (무거운 작업)")
        time.sleep(2)  # 로딩에 2초가 걸린다고 가정
        self.is_loaded = True
        print("모델 로딩 완료!")

    def predict(self, text: str) -> str:
        """
        실제 추론(Inference)을 수행하는 함수입니다.
        """
        if not self.is_loaded:
            raise RuntimeError("모델이 로드되지 않았습니다!")
        
        # 실제 AI 로직 대신 간단한 문자열 조작으로 결과를 냅니다.
        # 예: 입력된 텍스트 뒤에 " - AI 분석 완료"를 붙임
        result = f"{text} -> [감정: 긍정, 신뢰도: 99%]"
        return result