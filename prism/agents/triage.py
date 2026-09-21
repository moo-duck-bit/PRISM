"""Modality-Aware Triage (논문 3.2절).

클레임의 모달리티 수요를 미리 분류해 마스크 m을 만들고, 활성화할 전문 에이전트 집합
S(m)을 정한다. 명제를 전부 분해해 매번 모든 검증을 돌리는 방식과 달리,
비활성 에이전트는 그래프 수준에서 아예 실행되지 않는다(compute gating).

    tau(c) = m = (m_img, m_ent, m_time) in {0,1}^3
    S(m)   = {f} u {e | m_ent} u {st | m_time} u {v | m_img and |I| >= 1}

f(Factual)는 항상 활성이다. v(Visual)는 마스크가 켜져 있어도 실제 이미지가
없으면 활성화하지 않는다.
"""

import logging
import time
from typing import Dict, List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from prism.core.llm_models import get_llm
from prism.core.state import PrismState

logger = logging.getLogger(__name__)

parser = JsonOutputParser()

FACTUAL = "factual"
ENTITY = "entity"
SPATIOTEMPORAL = "spatiotemporal"
VISUAL = "visual"

# true/false 표기를 못박지 않으면 모델이 파이썬 표기(True/False)를 섞어 내보내 파싱이 깨진다.
PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a triage classifier for a multimodal fact-checking pipeline.
Read the claim and decide which specialist verifiers are needed.
Respond ONLY with valid JSON. Boolean values must be lowercase true or false.

1. "img": true if the claim refers to a photo or video, or if visual evidence is
   required to decide it.
2. "ent": true if the claim attributes a quote, action, position, or status to a
   named person or organization.
3. "time": true if the claim depends on a specific date, time, ordering, or
   duration of events.
4. "reasoning": one sentence explaining the assignment."""),
    ("user", "Claim: {claim}{context}"),
])

# 분류에 실패하면 검증을 넓게 여는 쪽으로 기운다. 놓치는 것보다 낫다.
FALLBACK_MASK = {"img": False, "ent": True, "time": True}


def active_agents(mask: Dict[str, bool], has_image: bool) -> List[str]:
    """S(m)을 계산한다. 그래프의 노드 이름 순서와 같은 순서로 돌려준다."""
    agents = [FACTUAL]
    if mask.get("ent"):
        agents.append(ENTITY)
    if mask.get("time"):
        agents.append(SPATIOTEMPORAL)
    if mask.get("img") and has_image:
        agents.append(VISUAL)
    return agents


def _context(state: PrismState) -> str:
    """포스트 텍스트와 OCR이 있으면 분류 근거로 함께 넘긴다."""
    parts = []
    if state.get("post_text"):
        parts.append(f"Post: {state['post_text']}")
    if state.get("ocr_text"):
        parts.append(f"OCR: {state['ocr_text']}")
    return ("\n" + "\n".join(parts)) if parts else ""


def triage_agent(state: PrismState) -> PrismState:
    """마스크 m과 활성 집합 S(m)을 상태에 기록한다."""
    started = time.time()

    try:
        chain = PROMPT | get_llm(task_type="triage") | parser
        raw = chain.invoke({"claim": state["claim"], "context": _context(state)})
    except Exception as exc:
        logger.warning("Triage 분류 실패, 기본 마스크 사용: %s", exc)
        raw = dict(FALLBACK_MASK, reasoning=f"triage failed: {exc}")

    mask = {key: bool(raw.get(key, FALLBACK_MASK[key])) for key in ("img", "ent", "time")}
    agents = active_agents(mask, has_image=bool(state.get("image_paths")))

    logger.info("Triage mask=%s -> active %d/4 %s", mask, len(agents), agents)

    metrics = state.get("metrics", {})
    metrics["triage_latency"] = round(time.time() - started, 4)
    metrics["active_agent_count"] = len(agents)
    for name in (FACTUAL, ENTITY, SPATIOTEMPORAL, VISUAL):
        metrics[f"active_{name}"] = int(name in agents)

    return {
        "modality_mask": mask,
        "active_agents": agents,
        "triage_rationale": str(raw.get("reasoning", "")),
        "metrics": metrics,
    }
