from langgraph.graph import StateGraph, END
from prism.core.state import PrismState

# 우리가 만든 모든 에이전트를 불러옵니다.
from prism.agents.triage import triage_agent
from prism.agents.retrieval_text import textual_retrieval_agent
from prism.agents.verifier_factual import factual_verifier
from prism.agents.verifier_entity import entity_verifier
from prism.agents.verifier_spatiotemporal import spatiotemporal_verifier
from prism.agents.explanation import explanation_agent

def build_prism_pipeline():
    """
    LangGraph를 사용하여 PRISM의 멀티 에이전트 파이프라인을 구축합니다.
    """
    print("🛠️ LangGraph 기반 PRISM 파이프라인 빌드 중...")
    
    # 1. 상태(State)를 공유하는 빈 그래프 생성
    workflow = StateGraph(PrismState)

    # 2. 노드(에이전트) 등록
    workflow.add_node("triage", triage_agent)
    workflow.add_node("retrieval_text", textual_retrieval_agent)
    workflow.add_node("verifier_factual", factual_verifier)
    workflow.add_node("verifier_entity", entity_verifier)
    workflow.add_node("verifier_spatiotemporal", spatiotemporal_verifier)
    workflow.add_node("explanation", explanation_agent)

    # 3. 고정된 흐름 연결 (시작 -> Triage -> 검색 -> 팩트 검증)
    workflow.set_entry_point("triage")
    workflow.add_edge("triage", "retrieval_text")
    workflow.add_edge("retrieval_text", "verifier_factual")

    # 4. 스마트 라우팅 함수 정의 (Triage 결과에 따라 길을 안내합니다)
    def route_after_factual(state: PrismState):
        results = state.get("triage_results", {})
        if results.get("entity"):
            return "verifier_entity"
        elif results.get("temporal"):
            return "verifier_spatiotemporal"
        else:
            return "explanation"

    def route_after_entity(state: PrismState):
        results = state.get("triage_results", {})
        if results.get("temporal"):
            return "verifier_spatiotemporal"
        else:
            return "explanation"

    # 5. 조건부 라우팅(Conditional Edges) 적용
    workflow.add_conditional_edges("verifier_factual", route_after_factual)
    workflow.add_conditional_edges("verifier_entity", route_after_entity)
    
    # 시공간 검증이 끝나면 무조건 설명 생성으로 갑니다.
    workflow.add_edge("verifier_spatiotemporal", "explanation")
    
    # 설명 생성이 끝나면 파이프라인 종료!
    workflow.add_edge("explanation", END)

    # 6. 그래프 컴파일 및 반환
    return workflow.compile()