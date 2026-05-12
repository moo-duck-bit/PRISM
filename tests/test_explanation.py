import os
import sys
import json
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from prism.core.state import PrismState
from prism.agents.explanation import explanation_agent

def run_explanation_test():
    load_dotenv()
    print("=== 📰 Explanation Agent (Toulmin) 단독 테스트 시작 ===\n")

    # 가상의 이전 단계 결과물들을 세팅합니다.
    mock_state: PrismState = {
        "claim": "A man was beaten to death outside Lauren Boebert's restaurant.",
        "triage_results": {},
        "text_evidence": [
            "Official autopsy reports show Anthony Royal Green died from a methamphetamine overdose.",
            "Police found no evidence of a physical beating leading to death."
        ],
        "image_evidence": [],
        "verdicts": {
            "factual": {
                "verdict": "Refuted",
                "confidence": 0.95,
                "rationale": "The autopsy clearly states overdose, not beating."
            }
        }
    }

    print("⏳ 에이전트가 툴민 구조로 기사를 작성 중입니다...")
    new_state = explanation_agent(mock_state)
    
    print("\n✅ [최종 작성된 팩트체크 설명문]")
    print(json.dumps(new_state.get("explanation", {}), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    run_explanation_test()