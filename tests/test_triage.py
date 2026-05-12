import os
import sys
import json
from dotenv import load_dotenv

# 프로젝트 최상단 경로를 시스템 경로에 추가하여 모듈을 임포트할 수 있게 합니다.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from prism.core.state import PrismState
from prism.core.data_loader import load_mocheg_data
from prism.agents.triage import triage_agent

def run_triage_test():
    # .env 파일에서 API 키 로드
    load_dotenv()
    print("=== 🚦 Triage 에이전트 단독 테스트 시작 ===\n")

    # 1. 실제 데이터 로드
    train_data = load_mocheg_data(split="train")
    
    # 첫 번째 데이터의 클레임을 가져옵니다.
    first_claim = train_data[0]["claim"]
    print(f"🕵️‍♂️ [분석할 클레임]:\n{first_claim}\n")

    # 2. 에이전트에 넘겨줄 초기 상태(State) 설정
    mock_state: PrismState = {
        "claim": first_claim,
        "triage_results": {},
        "text_evidence": [],
        "image_evidence": [],
        "verdicts": {}
    }

    # 3. Triage 에이전트 실행 (이 과정에서 LLM API가 호출됩니다)
    print("⏳ 에이전트가 클레임을 분석 중입니다...")
    new_state = triage_agent(mock_state)
    
    # 4. 결과 출력
    print("\n✅ [Triage 판단 결과]")
    # JSON 결과를 예쁘게(들여쓰기 2칸) 출력합니다.
    print(json.dumps(new_state["triage_results"], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    run_triage_test()