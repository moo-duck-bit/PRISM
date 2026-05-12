import os
import sys
import json
from dotenv import load_dotenv

# 파이썬 경로 설정 (프로젝트 최상단)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from prism.core.state import PrismState

# 우리가 피땀 흘려 만든 5개의 에이전트를 모두 불러옵니다!
from prism.agents.triage import triage_agent
from prism.agents.retrieval_text import textual_retrieval_agent
from prism.agents.verifier_factual import factual_verifier
from prism.agents.verifier_entity import entity_verifier
from prism.agents.verifier_spatiotemporal import spatiotemporal_verifier

def run_full_integration_test():
    load_dotenv()
    print("=== 🚀 PRISM 에이전트 릴레이 통합 테스트 시작 ===\n")

    # 1. 초기 상태 세팅 (시간, 장소, 인물이 모두 포함된 아주 복잡한 클레임)
    initial_claim = "A man was beaten to death outside Lauren Boebert's restaurant in Rifle, Colorado on a Friday night."
    
    state: PrismState = {
        "claim": initial_claim,
        "triage_results": {},
        "text_evidence": [],
        "image_evidence": [],
        "verdicts": {}
    }
    print(f"🕵️‍♂️ [분석할 클레임]: {state['claim']}\n")

    # ==========================================
    # STEP 1: 관제탑 (Triage)
    # ==========================================
    print("⏳ [STEP 1] Triage 에이전트 실행 중...")
    state.update(triage_agent(state))
    print(f"   👉 Triage 결과: {json.dumps(state['triage_results'], ensure_ascii=False)}\n")

    # ==========================================
    # STEP 2: 검색 (Retrieval)
    # ==========================================
    print("⏳ [STEP 2] Textual Retrieval 에이전트 실행 중...")
    state.update(textual_retrieval_agent(state))
    print(f"   👉 수집된 증거: {len(state['text_evidence'])}개")
    for i, ev in enumerate(state['text_evidence']):
        print(f"      - {i+1}: {ev}")
    print()

    # ==========================================
    # STEP 3: 검증 (Verification)
    # ==========================================
    print("⏳ [STEP 3] 검증 에이전트 릴레이 시작...\n")
    
    # 3-1. 사실 검증 (항상 실행)
    print("   🔍 Factual Verifier 실행 중...")
    state.update(factual_verifier(state))

    # 3-2. 인물 검증 (Triage 결과가 True일 때만)
    if state["triage_results"].get("entity", False):
        print("   👤 Triage 지시에 따라 Entity Verifier 실행 중...")
        state.update(entity_verifier(state))
    else:
        print("   ⏭️ Entity Verifier 건너뜀 (조건 불만족)")

    # 3-3. 시공간 검증 (Triage 결과가 True일 때만)
    if state["triage_results"].get("temporal", False):
        print("   ⏱️ Triage 지시에 따라 Spatiotemporal Verifier 실행 중...")
        state.update(spatiotemporal_verifier(state))
    else:
        print("   ⏭️ Spatiotemporal Verifier 건너뜀 (조건 불만족)")

    # ==========================================
    # 최종 결과 출력
    # ==========================================
    print("\n✅ [통합 테스트 최종 판정 내역 (Verdicts)]")
    print(json.dumps(state["verdicts"], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    run_full_integration_test()