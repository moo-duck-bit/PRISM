"""LangGraph 파이프라인 조립.

    triage -> retrieve(E_T, E_I) -> S(m) 에 속한 검증기만 -> calibrate -> explain

Triage의 게이팅은 그래프 수준에서 일어난다. 비활성 에이전트는 프롬프트 안에서
건너뛰는 것이 아니라 노드 자체가 실행되지 않는다(compute gating).

ablation 스위치:
    use_triage=False       마스크를 전부 켠다. 4개 에이전트가 모두 실행된다.
    evidence_gating=False  E_T 를 비운다(closed-book).
    use_calibrator=False   delta_theta 와 psi_guard 를 끄고 LLM 출력을 그대로 쓴다.
"""

from functools import partial
from typing import Dict, List

from langgraph.graph import END, StateGraph

from prism.agents.generator_toulmin import toulmin_generator_agent
from prism.agents.retrieval_text import textual_retrieval_agent
from prism.agents.retrieval_visual import visual_retrieval_agent
from prism.agents.triage import active_agents, triage_agent
from prism.agents.verifier_entity import entity_verifier
from prism.agents.verifier_factual import factual_verifier
from prism.agents.verifier_spatiotemporal import spatiotemporal_verifier
from prism.agents.verifier_visual import visual_verifier
from prism.core.calibrator import calibrate
from prism.core.state import PrismState

# 그래프 노드 이름과 S(m) 원소의 대응. 순서가 곧 실행 순서다.
VERIFIER_NODES: List[tuple] = [
    ("verify_factual", "factual", factual_verifier),
    ("verify_entity", "entity", entity_verifier),
    ("verify_spatiotemporal", "spatiotemporal", spatiotemporal_verifier),
    ("verify_visual", "visual", visual_verifier),
]

CALIBRATE = "calibrate"
EXPLAIN = "explain"

ALL_MODALITIES = {"img": True, "ent": True, "time": True}


def _bypass_triage(state: PrismState) -> PrismState:
    """Triage를 제거한 ablation 경로. 모든 검증기를 활성화한다."""
    mask = dict(ALL_MODALITIES)
    agents = active_agents(mask, has_image=bool(state.get("image_paths")))

    metrics = state.get("metrics", {})
    metrics["triage_latency"] = 0.0
    metrics["active_agent_count"] = len(agents)
    for _, agent, _fn in VERIFIER_NODES:
        metrics[f"active_{agent}"] = int(agent in agents)

    return {
        "modality_mask": mask,
        "active_agents": agents,
        "triage_rationale": "triage disabled",
        "metrics": metrics,
    }


def _calibrate_node(state: PrismState, use_rules: bool) -> PrismState:
    return calibrate(state, use_rules=use_rules)


def _router(after_index: int, terminal: str):
    """after_index 다음으로 실행할 활성 검증기를 고른다. 없으면 terminal로 간다."""

    def route(state: PrismState) -> str:
        active = set(state.get("active_agents", []))
        for node, agent, _fn in VERIFIER_NODES[after_index + 1:]:
            if agent in active:
                return node
        return terminal

    return route


def _route_targets(after_index: int, terminal: str) -> List[str]:
    targets = [node for node, _a, _f in VERIFIER_NODES[after_index + 1:]]
    return targets + [terminal]


def build_graph(
    evidence_gating: bool = True,
    use_triage: bool = True,
    use_calibrator: bool = True,
    generate_explanation: bool = True,
):
    """PRISM 파이프라인을 컴파일한다.

    Args:
        evidence_gating: False면 외부 증거 없이 판단한다(closed-book).
        use_triage: False면 마스크를 전부 켜고 4개 에이전트를 모두 실행한다.
        use_calibrator: False면 결정적 규칙과 가드레일을 끄고 LLM 출력을 그대로 쓴다.
        generate_explanation: False면 Toulmin 생성 노드를 뺀다. 판정만 측정할 때 쓴다.
    """
    workflow = StateGraph(PrismState)

    workflow.add_node("triage", triage_agent if use_triage else _bypass_triage)
    workflow.add_node("retrieve_text", partial(textual_retrieval_agent, gated=evidence_gating))
    workflow.add_node("retrieve_visual", visual_retrieval_agent)

    for node, _agent, fn in VERIFIER_NODES:
        workflow.add_node(node, fn)

    workflow.add_node(CALIBRATE, partial(_calibrate_node, use_rules=use_calibrator))

    terminal = EXPLAIN if generate_explanation else END
    if generate_explanation:
        workflow.add_node(EXPLAIN, toulmin_generator_agent)

    workflow.set_entry_point("triage")
    workflow.add_edge("triage", "retrieve_text")
    workflow.add_edge("retrieve_text", "retrieve_visual")

    # f는 항상 활성이므로 첫 검증기로 직행한다.
    workflow.add_edge("retrieve_visual", VERIFIER_NODES[0][0])

    # 각 검증기 뒤에서 다음 활성 검증기로 건너뛴다. 비활성 노드는 실행되지 않는다.
    for index, (node, _agent, _fn) in enumerate(VERIFIER_NODES[:-1]):
        workflow.add_conditional_edges(
            node, _router(index, CALIBRATE), _route_targets(index, CALIBRATE)
        )
    workflow.add_edge(VERIFIER_NODES[-1][0], CALIBRATE)

    workflow.add_edge(CALIBRATE, terminal)
    if generate_explanation:
        workflow.add_edge(EXPLAIN, END)

    return workflow.compile()


ABLATIONS: Dict[str, Dict[str, bool]] = {
    "full": {},
    "no_gating": {"evidence_gating": False},
    "no_triage": {"use_triage": False},
    "no_calibrator": {"use_calibrator": False},
    "no_orchestration": {"use_triage": False, "use_calibrator": False},
}


def build_variant(name: str = "full", **overrides):
    """ablation 이름으로 파이프라인을 만든다."""
    try:
        options = dict(ABLATIONS[name])
    except KeyError:
        raise ValueError(
            f"알 수 없는 ablation: {name!r}. 사용 가능: {', '.join(ABLATIONS)}"
        ) from None
    options.update(overrides)
    return build_graph(**options)


def initial_state(
    claim: str,
    claim_id: str = "",
    post_text: str = "",
    ocr_text: str = "",
    image_paths: List[str] = None,
    evidence_pool: List[str] = None,
) -> PrismState:
    """입력 x = (c, P, O, I) 로 초기 상태를 만든다."""
    return {
        "claim_id": claim_id,
        "claim": claim,
        "post_text": post_text,
        "ocr_text": ocr_text,
        "image_paths": list(image_paths or []),
        "evidence_pool": list(evidence_pool or []),
        "modality_mask": {},
        "active_agents": [],
        "text_evidence": [],
        "image_evidence": [],
        "verdicts": {},
        "verdict": "",
        "calibration": {},
        "explanation": {},
        "report": "",
        "metrics": {},
    }
