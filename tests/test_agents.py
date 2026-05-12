import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from prism.core.state import PrismState
from prism.core.data_loader import load_mocheg_data
from prism.agents.triage import triage_agent

def run_tests():
    load_dotenv()
    print("=== PRISM Triage 에이전트 테스트 시작 ===\n")

    # 1. 아까 성공했던 데이터 로더로 첫 번째 데이터를 가져옵니다.
    train_data = load_mocheg_data(split="train")
    first_claim = train_data[0]["claim"]
    
    print(f"\n[분석할 클레임]: {first_claim}")

    # 2. 초기 상태(State) 설정
    mock_state: PrismState = {
        "claim": first_claim,
        "triage_results": {},
        "text_evidence": [],
        "image_evidence": [],
        "verdicts": {}
    }

    # 3. Triage 에이전트 실행
    new_state = triage_agent(mock_state)
    
    print("\n[Triage 결과]")
    print(new_state["triage_results"])

if __name__ == "__main__":
    run_tests()