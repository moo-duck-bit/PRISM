import os
import sys
import json
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from prism.core.state import PrismState
from prism.agents.verifier_spatiotemporal import spatiotemporal_verifier

def run_spatiotemporal_test():
    load_dotenv()
    print("=== ⏱️ Spatiotemporal Verifier 단독 테스트 시작 ===\n")

    # 시간과 장소가 명시된 가상의 클레임과 증거를 주입합니다.
    mock_state: PrismState = {
        "claim": "A man was beaten to death outside Lauren Boebert's restaurant in Rifle, Colorado on a Friday night.",
        "text_evidence": [
            "The incident occurred near Shooters Grill in Rifle, Colorado.",
            "Police records show the event happened on a Wednesday morning, not Friday night."
        ],
        "verdicts": {}
    }

    print("⏳ 시공간 일치 여부 검증 중...")
    new_state = spatiotemporal_verifier(mock_state)
    
    print("\n✅ [Spatiotemporal 판정 결과]")
    print(json.dumps(new_state["verdicts"]["spatiotemporal"], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    run_spatiotemporal_test()