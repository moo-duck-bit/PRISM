"""Decoupled Explanation via Toulmin Representation (논문 3.5절).

    E = g_Toulmin(c, {r_s}, y_hat)

생성기는 동결된 rationale {r_s} 와 확정된 판정 y_hat 만 받는다. 판정을 다시 계산하지
않으므로, 설명 단계의 환각이 분류를 오염시키는 경로가 구조적으로 막힌다.
판정 경로가 temperature 0으로 고정된 뒤에만 호출되며, 이 생성기만 temperature 0.3을 쓴다.

출력은 Toulmin 6요소로 고정된다. 자유 서술과 달리 요소 누락 여부를 기계적으로
확인할 수 있어, 구조적 완전성이 생성 결과에 좌우되지 않는다.
"""

import logging
import time
from typing import Any, Dict

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from prism.core.llm_models import get_llm
from prism.core.state import PrismState

logger = logging.getLogger(__name__)

parser = JsonOutputParser()

TOULMIN_ELEMENTS = ("claim", "grounds", "warrant", "backing", "qualifier", "rebuttal")

ELEMENT_LABELS = {
    "claim": "주장",
    "grounds": "근거",
    "warrant": "보증",
    "backing": "뒷받침",
    "qualifier": "한정",
    "rebuttal": "반박",
}

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are writing the final report of a fact-checking system.

The verdict has already been decided and is final. Your task is to externalize
the reasoning behind it into the six elements of the Toulmin model. You must not
re-evaluate, question, or contradict the verdict.

You are given only the verifier rationales. Every statement you write must trace
back to one of them. If an element is not supported by any rationale, say so
briefly rather than inventing content.

Respond ONLY with valid JSON using these six keys:
1. "claim": the proposition that was checked, restated plainly.
2. "grounds": the evidence the verifiers actually cited.
3. "warrant": why that evidence bears on the claim.
4. "backing": what gives the evidence its standing (source, scope, verifier).
5. "qualifier": the strength and limits of the conclusion, consistent with the
   given verdict. If the verdict is NEI, state what evidence was missing.
6. "rebuttal": conditions or conflicting signals under which the conclusion would
   not hold. Write "해당 없음" if the rationales report none.

Write the values in Korean."""),
    ("user", "Verdict (final, do not change): {verdict}\n\n"
             "Claim: {claim}\n\nVerifier rationales:\n{rationales}"),
])


def _rationales(state: PrismState) -> str:
    lines = []
    for name, result in state.get("verdicts", {}).items():
        lines.append(f"- [{name}] {result.get('verdict')} :: {result.get('rationale')}")
    return "\n".join(lines) if lines else "- (no verifier rationale available)"


def render_report(verdict: str, elements: Dict[str, Any]) -> str:
    """6요소를 사람이 읽을 형태로 렌더링한다. LLM을 다시 부르지 않는다."""
    lines = [f"판정: {verdict}", ""]
    for key in TOULMIN_ELEMENTS:
        value = str(elements.get(key, "")).strip() or "(누락)"
        lines.append(f"[{ELEMENT_LABELS[key]}] {value}")
    return "\n".join(lines)


def structural_completeness(elements: Dict[str, Any]) -> float:
    """6요소 중 실제로 채워진 비율. 자유 서술과의 비교 지표로 쓴다."""
    filled = sum(1 for key in TOULMIN_ELEMENTS if str(elements.get(key, "")).strip())
    return filled / len(TOULMIN_ELEMENTS)


def toulmin_generator_agent(state: PrismState) -> PrismState:
    started = time.time()
    verdict = state.get("verdict", "")

    try:
        chain = PROMPT | get_llm(task_type="explanation") | parser
        raw = chain.invoke({
            "verdict": verdict,
            "claim": state.get("claim", ""),
            "rationales": _rationales(state),
        })
        elements = {key: str(raw.get(key, "")).strip() for key in TOULMIN_ELEMENTS}
    except Exception as exc:
        logger.error("설명 생성 실패: %s", exc)
        elements = {key: "" for key in TOULMIN_ELEMENTS}

    metrics = state.get("metrics", {})
    metrics["generator_latency"] = round(time.time() - started, 4)
    metrics["structural_completeness"] = structural_completeness(elements)

    return {
        "explanation": elements,
        "report": render_report(verdict, elements),
        "metrics": metrics,
    }
