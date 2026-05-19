"""
graph_aggressive.py — Anti-NEI 강화 버전 파이프라인

[기존 graph.py 와의 차이점]
- 4개 Verifier 모두 *_aggressive 버전으로 교체
- Toulmin Generator는 동일 (기사 생성 목적이므로 판정 로직 불변)

비교 실험:
  - 기존 (보수적): from prism.core.graph import prism_app
  - Aggressive    : from prism.core.graph_aggressive import prism_app_aggressive
"""
from langgraph.graph import StateGraph, END
from prism.core.state import PrismState

# ── 기존 에이전트 (변경 없음) ──────────────────────────────────────────────────
from prism.agents.retrieval_text import textual_retrieval_agent
from prism.agents.generator_toulmin import toulmin_generator_agent

# ── Aggressive 버전 Verifier ──────────────────────────────────────────────────
from prism.agents.verifier_factual_aggressive import factual_verifier_aggressive
from prism.agents.verifier_entity_aggressive import entity_verifier_aggressive
from prism.agents.verifier_spatiotemporal_aggressive import spatiotemporal_verifier_aggressive
from prism.agents.verifier_visual_aggressive import visual_verifier_aggressive

print("⚙️ [LangGraph] PRISM Aggressive (Anti-NEI) 파이프라인 조립 중...")

workflow_aggressive = StateGraph(PrismState)

workflow_aggressive.add_node("retrieve",                textual_retrieval_agent)
workflow_aggressive.add_node("verify_factual",          factual_verifier_aggressive)
workflow_aggressive.add_node("verify_entity",           entity_verifier_aggressive)
workflow_aggressive.add_node("verify_spatiotemporal",   spatiotemporal_verifier_aggressive)
workflow_aggressive.add_node("verify_visual",           visual_verifier_aggressive)
workflow_aggressive.add_node("generate_article",        toulmin_generator_agent)

workflow_aggressive.set_entry_point("retrieve")
workflow_aggressive.add_edge("retrieve",              "verify_factual")
workflow_aggressive.add_edge("verify_factual",        "verify_entity")
workflow_aggressive.add_edge("verify_entity",         "verify_spatiotemporal")
workflow_aggressive.add_edge("verify_spatiotemporal", "verify_visual")
workflow_aggressive.add_edge("verify_visual",         "generate_article")
workflow_aggressive.add_edge("generate_article",      END)

prism_app_aggressive = workflow_aggressive.compile()
print("✅ [LangGraph] Aggressive 파이프라인 조립 완료!")
