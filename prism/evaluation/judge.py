"""설명 품질의 자동 평가.

두 지표를 따로 본다.

Faithfulness
    설명을 원자 명제로 분해한 뒤, 각 명제가 증거 집합
    E = E_T u E_I u {r_s} 에 의해 함의되는 비율.
    정답 라벨이 아니라 증거를 기준으로 본다.

Decision-Fidelity
    설명의 결론이 시스템 판정 y_hat 과 일치하는 비율.
    정답이 아니라 y_hat 과 비교한다는 점이 핵심이다. 판정과 설명이 분리되어 있다면
    설명은 정답이 아니라 판정을 따라가야 하고, 이 값이 1에 가까워야 한다.

둘 다 temperature 0으로 채점한다.
"""

import json
import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from prism.core.llm_models import get_llm
from prism.core.verdict import NEI, REFUTED, SUPPORTED

logger = logging.getLogger(__name__)

FAITHFULNESS_PROMPT = """You are evaluating whether an explanation stays inside its evidence.

Decompose the explanation into atomic propositions: short, independently
checkable factual statements. Ignore hedges, connectives, and restatements of
the verdict itself.

For each proposition, decide whether the evidence set entails it. A proposition
is entailed only if the evidence states or directly implies it. Plausibility,
world knowledge, and common sense do not count as entailment.

Respond ONLY with valid JSON:
{"propositions": [{"text": "...", "entailed": true}, ...]}"""

FIDELITY_PROMPT = """You are checking whether an explanation agrees with a verdict.

You are given a system verdict and an explanation. Decide whether the conclusion
the explanation reaches is the same as the system verdict. Judge agreement with
the verdict only. Do not consider whether the verdict is correct.

Respond ONLY with valid JSON:
{"conclusion": "Supported" | "Refuted" | "NEI", "agrees": true}"""

FAILED = {"faithfulness": 0.0, "n_propositions": 0, "decision_fidelity": 0}


def _parse(content: str) -> Dict[str, Any]:
    return json.loads(content.replace("```json", "").replace("```", "").strip())


def _evidence_block(text_evidence: List[str], rationales: List[str], has_image: bool) -> str:
    parts = list(text_evidence)
    parts += [f"[verifier rationale] {r}" for r in rationales if r]
    if has_image:
        parts.append("[image] An image was supplied and analyzed by the visual verifier.")
    return "\n".join(parts) if parts else "(no evidence)"


def faithfulness(explanation: str, text_evidence: List[str], rationales: List[str],
                 has_image: bool = False) -> Dict[str, Any]:
    """증거에 의해 함의되는 원자 명제의 비율."""
    user = (
        f"Evidence set:\n{_evidence_block(text_evidence, rationales, has_image)}\n\n"
        f"Explanation:\n{explanation}"
    )
    try:
        response = get_llm(task_type="judge").invoke([
            SystemMessage(content=FAITHFULNESS_PROMPT),
            HumanMessage(content=user),
        ])
        propositions = _parse(response.content).get("propositions", [])
    except Exception as exc:
        logger.warning("Faithfulness 채점 실패: %s", exc)
        return {"faithfulness": 0.0, "n_propositions": 0}

    if not propositions:
        return {"faithfulness": 0.0, "n_propositions": 0}

    entailed = sum(1 for p in propositions if p.get("entailed"))
    return {
        "faithfulness": entailed / len(propositions),
        "n_propositions": len(propositions),
    }


def decision_fidelity(explanation: str, verdict: str) -> int:
    """설명의 결론이 시스템 판정과 일치하면 1."""
    try:
        response = get_llm(task_type="judge").invoke([
            SystemMessage(content=FIDELITY_PROMPT),
            HumanMessage(content=f"System verdict: {verdict}\n\nExplanation:\n{explanation}"),
        ])
        payload = _parse(response.content)
    except Exception as exc:
        logger.warning("Decision-Fidelity 채점 실패: %s", exc)
        return 0

    conclusion = str(payload.get("conclusion", "")).strip().lower()
    mapping = {"supported": SUPPORTED, "refuted": REFUTED, "nei": NEI}
    return int(mapping.get(conclusion) == verdict and bool(payload.get("agrees", True)))


def score_explanation(
    explanation: str,
    verdict: str,
    text_evidence: List[str],
    rationales: List[str],
    has_image: bool = False,
) -> Dict[str, Any]:
    """두 지표를 한 번에 채점한다. 실패해도 실행은 계속한다."""
    if not explanation.strip():
        return dict(FAILED)

    scores = faithfulness(explanation, text_evidence, rationales, has_image)
    scores["decision_fidelity"] = decision_fidelity(explanation, verdict)
    return scores
