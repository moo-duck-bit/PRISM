import os
import sys

# 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from prism.agents.retrieval_text import textual_retrieval_agent
from prism.agents.verifier_factual import factual_verifier
from prism.agents.verifier_entity import entity_verifier
from prism.agents.verifier_spatiotemporal import spatiotemporal_verifier
from prism.agents.generator_toulmin import toulmin_generator_agent
from prism.core.state import PrismState

def run_full_text_pipeline():
    print("=== 🌟 PRISM 텍스트 파이프라인 마스터 테스트 시작 ===\n")
    
    # 1. 초기 상태 설정 (검증이 모두 필요한 상황 가정)
    state: PrismState = {
        "claim": "Ellen Weintraub talks about voter fraud.",
        "triage_results": {
            "factual_needed": True, 
            "entity_needed": True, 
            "spatiotemporal_needed": True
        },
        "text_evidence": [],
        "image_evidence": [],
        "verdicts": {},
        "final_article": ""
    }
    
    # 2. 검색 에이전트 (BM25)
    state.update(textual_retrieval_agent(state))
    
    # 3. 3대 텍스트 검증기 가동 (Multi-Agent Verification)
    state.update(factual_verifier(state))
    state.update(entity_verifier(state))
    state.update(spatiotemporal_verifier(state))
    
    # 4. 최종 툴민(Toulmin) 기사 작성
    state.update(toulmin_generator_agent(state))
    
    print("\n" + "="*50)
    print("📜 [최종 생성된 팩트체킹 보고서]")
    print("="*50)
    print(state.get("final_article", "기사 생성 실패"))
    print("="*50)

if __name__ == "__main__":
    run_full_text_pipeline()