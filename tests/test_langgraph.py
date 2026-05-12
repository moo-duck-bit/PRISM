import os
import sys
import json
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from prism.core.state import PrismState
from prism.core.pipeline import build_prism_pipeline

def run_langgraph_test():
    load_dotenv()
    print("=== 🚀 PRISM LangGraph 풀 파이프라인 테스트 시작 ===\n")

    # 파이프라인 조립
    app = build_prism_pipeline()

    initial_claim = "A man was beaten to death outside Lauren Boebert's restaurant in Rifle, Colorado on a Friday night."
    
    # 초기 상태
    initial_state: PrismState = {
        "claim": initial_claim,
        "triage_results": {},
        "text_evidence": [],
        "image_evidence": [],
        "verdicts": {},
        "explanation": {}  
    }

    print(f"\n🕵️‍♂️ [분석할 클레임]: {initial_claim}\n")
    print("⏳ 파이프라인 실행 중... (LangGraph가 에이전트들을 지휘합니다)\n")
    
    # LangGraph 실행 (invoke 한 번으로 끝까지 돌아갑니다!)
    final_state = app.invoke(initial_state)

    print("\n=======================================================")
    print("✅ [LangGraph 최종 판정 요약 (Verdicts)]")
    for agent_name, result in final_state.get("verdicts", {}).items():
        print(f" - {agent_name.capitalize()}: {result.get('verdict')} (확신도: {result.get('confidence')})")

    print("\n📰 [최종 설명문 (Toulmin Structure)]")
    print(json.dumps(final_state.get("explanation", {}), indent=2, ensure_ascii=False))
    print("=======================================================\n")

if __name__ == "__main__":
    run_langgraph_test()