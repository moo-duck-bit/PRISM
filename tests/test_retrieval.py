import os
import sys
import json
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from prism.core.state import PrismState
from prism.agents.triage import triage_agent
from prism.agents.retrieval_text import textual_retrieval_agent
from prism.agents.retrieval_visual import visual_retrieval_agent

def run_triage_and_retrieval_test():
    load_dotenv()
    print("=== 🔗 Triage -> Retrieval 통합 테스트 시작 ===\n")

    state: PrismState = {
        "claim": "A man was beaten to death outside Lauren Boebert's restaurant.",
        "triage_results": {},
        "text_evidence": [],
        "image_evidence": [],
        "verdicts": {}
    }
    
    print("⏳ 1. Triage 에이전트 실행...")
    state.update(triage_agent(state))
    print(f"   [Triage 판단]: image_needed={state['triage_results'].get('image_needed')}\n")

    print("⏳ 2. Textual Retrieval 에이전트 실행...")
    state.update(textual_retrieval_agent(state))
    print(f"   [텍스트 증거 수집 완료]: {len(state['text_evidence'])}개")
    for i, ev in enumerate(state['text_evidence']):
        print(f"      - 증거 {i+1}: {ev}")
    print()

    print("⏳ 3. Visual Retrieval 에이전트 실행...")
    state.update(visual_retrieval_agent(state))
    print(f"   [이미지 증거 수집 완료]: {len(state['image_evidence'])}개\n")

    print("✅ [최종 상태(State) 점검]")
    print(json.dumps({
        "triage_results": state["triage_results"],
        "text_evidence_count": len(state["text_evidence"]),
        "image_evidence_count": len(state["image_evidence"])
    }, indent=2))

if __name__ == "__main__":
    run_triage_and_retrieval_test()