from langgraph.graph import StateGraph, END
from prism.core.state import PrismState

# 1. 모든 에이전트 불러오기
from prism.agents.retrieval_text import textual_retrieval_agent
from prism.agents.verifier_factual import factual_verifier
from prism.agents.verifier_entity import entity_verifier
from prism.agents.verifier_spatiotemporal import spatiotemporal_verifier
from prism.agents.verifier_visual import visual_verifier  # 👁️ 시각 검증기 추가!
from prism.agents.generator_toulmin import toulmin_generator_agent

print("⚙️ [LangGraph] PRISM 멀티모달 파이프라인 조립 중...")

# 2. 그래프 뼈대 생성
workflow = StateGraph(PrismState)

# 3. 노드(공장) 등록
workflow.add_node("retrieve", textual_retrieval_agent)
workflow.add_node("verify_factual", factual_verifier)
workflow.add_node("verify_entity", entity_verifier)
workflow.add_node("verify_spatiotemporal", spatiotemporal_verifier)
workflow.add_node("verify_visual", visual_verifier)       # 👁️ 시각 공장 등록!
workflow.add_node("generate_article", toulmin_generator_agent)

# 4. 엣지(컨베이어 벨트) 연결 (순서대로 쫙 흘러갑니다)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "verify_factual")
workflow.add_edge("verify_factual", "verify_entity")
workflow.add_edge("verify_entity", "verify_spatiotemporal")
workflow.add_edge("verify_spatiotemporal", "verify_visual") # 시공간 다음엔 눈으로 확인!
workflow.add_edge("verify_visual", "generate_article")      # 눈으로 본 결과까지 모아서 기사 작성!
workflow.add_edge("generate_article", END)

# 5. 엔진 컴파일
prism_app = workflow.compile()
print("✅ [LangGraph] 멀티모달 조립 완료! 시스템 가동 준비 끝.")