import os
import sys
import json
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from prism.core.state import PrismState
from prism.agents.verifier_entity import entity_verifier

def run_entity_test():
    load_dotenv()
    print("=== 👤 Entity Verifier 단독 테스트 시작 ===\n")

    # 인물(Entity)에 대한 연관성을 확인할 수 있는 모의 데이터를 넣습니다.
    mock_state: PrismState = {
        "claim": "A man was beaten to death outside Lauren Boebert's restaurant.",
        "text_evidence": [
            "Official autopsy reports show Anthony Royal Green died from a methamphetamine overdose.",
            "Lauren Boebert is the owner of Shooters Grill, but the incident was an overdose, not a beating."
        ],
        "verdicts": {}
    }

    print("⏳ 인물 및 기관 연관성 검증 중...")
    new_state = entity_verifier(mock_state)
    
    print("\n✅ [Entity 판정 결과]")
    print(json.dumps(new_state["verdicts"]["entity"], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    run_entity_test()