"""Deterministic-First Cross-Modal Calibration (논문 3.4절).

검증기 판정 {v_s}를 최종 판정 y_hat 하나로 수렴시킨다. 규칙을 먼저 적용하고,
규칙이 해소하지 못한 경우(bottom)만 LLM에 위임한다. LLM 출력에는 다시 가드레일을 씌운다.

    y_hat = delta_theta(sigma)                     if delta_theta(sigma) != bottom
          = psi_guard(LLM(sigma, {v_s}))           otherwise

두 경로로 동작한다.
(a) 멀티모달 경로: 이미지가 분석된 경우. 교차모달 규칙을 먼저 적용한다.
(b) 텍스트 우세 경로: 정족수와 기권 가드를 LLM 출력에 적용한다.
    NEI 기권 능력은 주로 이 경로에서 나온다.

판정 경로의 모든 호출은 temperature 0으로 돌아 결과가 결정적이다.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from prism.core.llm_models import get_llm
from prism.core.state import PrismState
from prism.core.verdict import (
    NEI,
    REFUTED,
    SUPPORTED,
    TEXT_VERIFIERS,
    VISUAL_VERIFIER,
    majority_label,
    quorum_label,
    vote_counts,
)

logger = logging.getLogger(__name__)

# 규칙이 판단을 유보했음을 나타내는 값. 논문의 bottom 기호에 해당한다.
BOTTOM = None


@dataclass(frozen=True)
class Theta:
    """theta = (tau_v, tau_q). 정치 도메인 held-out dev split에서 고른 값이다.

    tau_v는 교차모달 충돌이 일어난 소수 사례에만 적용되며,
    [0.6, 0.8] 구간에서 결과가 안정적이다.
    """

    tau_v: float = 0.7   # 시각 판정을 신뢰할 최소 confidence
    tau_q: int = 3       # 합의 정족수


DEFAULT_THETA = Theta()

_SYSTEM_PROMPT = """You are the verdict calibrator of a fact-checking system.
You are given a claim and the verdicts of several specialist verifiers.
Each verifier reports a verdict, a confidence, and a rationale for its own scope only.

Decide the final verdict. Rules:
1. Weigh verifiers by confidence and by how directly their scope bears on the claim.
2. "NEI" means the available evidence is insufficient to decide. It is a valid
   verdict, not a failure. Choose it when the verifiers do not converge and no
   side is grounded in the evidence.
3. Do not introduce facts that no verifier reported.

Respond ONLY with valid JSON:
{"verdict": "Supported" | "Refuted" | "NEI", "rationale": "one short sentence"}"""


def _visual(state: PrismState) -> Optional[Dict[str, Any]]:
    result = state.get("verdicts", {}).get(VISUAL_VERIFIER)
    return result if isinstance(result, dict) else None


def has_image_analysis(state: PrismState) -> bool:
    """멀티모달 경로 여부. 시각 검증기가 실제로 돌았을 때만 참이다."""
    return _visual(state) is not None and bool(state.get("image_evidence"))


def conflict(state: PrismState) -> bool:
    """교차모달 충돌 판정.

    conflict(sigma) = (y_v = Refuted) and (n_sup_txt >= 2)
                      and (n_ref_txt = 0) and (E_I != empty)

    텍스트 검증기들이 일치해서 Supported로 기울었는데 시각 증거만 반박하는,
    가장 위험한 형태의 불일치다.
    """
    visual = _visual(state)
    if visual is None or visual.get("verdict") != REFUTED:
        return False
    if not state.get("image_evidence"):
        return False

    counts = vote_counts(state.get("verdicts", {}), TEXT_VERIFIERS)
    return counts[SUPPORTED] >= 2 and counts[REFUTED] == 0


def _modality_conflict(state: PrismState) -> bool:
    """시각 판정과 텍스트 다수 의견이 정면으로 어긋나는 경우.

    conflict()가 잡는 패턴(텍스트 2표 이상 Supported + 시각 Refuted)에는
    해당하지 않지만, 모달리티 사이에 결정적 불일치가 있는 상황이다.
    """
    visual = _visual(state)
    if visual is None:
        return False

    visual_verdict = visual.get("verdict")
    if visual_verdict == NEI:
        return False

    text_majority = majority_label(vote_counts(state.get("verdicts", {}), TEXT_VERIFIERS))
    if text_majority is None or text_majority == NEI:
        return False

    return text_majority != visual_verdict


def delta_theta(state: PrismState, theta: Theta = DEFAULT_THETA) -> Optional[str]:
    """결정적 규칙 delta_theta. 적용 가능한 규칙이 없으면 BOTTOM을 돌려준다.

    규칙 적용 순서는 안전성 우선이다. 교차모달 충돌 규칙을 정족수보다 먼저 보는 이유는,
    텍스트 검증기가 모두 Supported로 모여 정족수를 채운 상황이야말로
    위험한 오탐(Refuted를 Supported로 판정)이 발생하는 지점이기 때문이다.
    """
    # (ii) 충돌 시 시각 confidence 게이팅
    if conflict(state):
        visual = _visual(state)
        if visual["confidence"] >= theta.tau_v:
            return REFUTED
        return NEI

    counts = vote_counts(state.get("verdicts", {}))

    # (i) 합의 정족수
    label = quorum_label(counts, theta.tau_q)
    if label is not None:
        return label

    # (iii) 교차모달 또는 모달리티 불일치는 기권한다
    if _modality_conflict(state):
        return NEI

    return BOTTOM


def psi_guard(
    llm_verdict: str,
    state: PrismState,
    theta: Theta = DEFAULT_THETA,
) -> Tuple[str, str]:
    """LLM 출력에 검증기 합의를 강제하는 가드레일.

    1. 기권 가드: 충돌 상황인데 LLM이 Supported를 냈다면 시각 판정 또는 NEI로 교정한다.
       Refuted를 Supported로 올리는 승격은 허용하지 않는다.
    2. 정족수 승격: 정족수를 채운 라벨이 있으면 그 라벨로 확정한다.
    3. 과반 승격: 과반 라벨이 있으면 그 라벨로 확정한다.

    반환값은 (판정, 적용된 규칙 이름)이다.
    """
    if conflict(state) and llm_verdict == SUPPORTED:
        visual = _visual(state)
        corrected = REFUTED if visual["confidence"] >= theta.tau_v else NEI
        logger.info("psi_guard: R->S 승격 차단, %s 로 교정", corrected)
        return corrected, "guard_block_rs"

    counts = vote_counts(state.get("verdicts", {}))

    label = quorum_label(counts, theta.tau_q)
    if label is not None and label != llm_verdict:
        return label, "guard_quorum"
    if label is not None:
        return label, "llm_agrees_quorum"

    label = majority_label(counts)
    if label is not None and label != llm_verdict:
        return label, "guard_majority"
    if label is not None:
        return label, "llm_agrees_majority"

    return llm_verdict, "llm"


def _llm_verdict(state: PrismState) -> Tuple[str, str]:
    """규칙이 해소하지 못한 경우에만 호출한다."""
    lines = []
    for name, result in state.get("verdicts", {}).items():
        lines.append(
            f"- {name}: {result.get('verdict')} "
            f"(confidence {result.get('confidence')}) :: {result.get('rationale')}"
        )
    user = f"Claim: {state.get('claim', '')}\n\nVerifier verdicts:\n" + "\n".join(lines)

    try:
        response = get_llm(task_type="calibration").invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user),
        ])
        payload = json.loads(response.content.replace("```json", "").replace("```", "").strip())
        raw = str(payload.get("verdict", "")).strip().lower()
    except Exception as exc:
        logger.warning("캘리브레이션 LLM 호출 실패: %s", exc)
        return NEI, f"llm error: {exc}"

    mapping = {"supported": SUPPORTED, "refuted": REFUTED, "nei": NEI}
    return mapping.get(raw, NEI), str(payload.get("rationale", ""))


def calibrate(
    state: PrismState,
    theta: Theta = DEFAULT_THETA,
    use_rules: bool = True,
) -> Dict[str, Any]:
    """최종 판정 y_hat과 그 근거 기록을 만든다.

    Args:
        use_rules: False면 규칙을 전부 끄고 LLM 출력을 그대로 쓴다.
            캘리브레이터 제거 ablation에 해당하며, 이 경우 기권 능력이 사라진다.
    """
    trace: Dict[str, Any] = {
        "path": "multimodal" if has_image_analysis(state) else "text_dominant",
        "conflict": conflict(state),
        "theta": {"tau_v": theta.tau_v, "tau_q": theta.tau_q},
    }

    if not use_rules:
        verdict, rationale = _llm_verdict(state)
        trace.update(rule="disabled", llm_verdict=verdict, llm_rationale=rationale)
        return {"verdict": verdict, "calibration": trace}

    # (a) 멀티모달 경로: 결정적 규칙을 먼저 적용한다
    if trace["path"] == "multimodal":
        deterministic = delta_theta(state, theta)
        if deterministic is not BOTTOM:
            trace.update(rule="delta_theta", deterministic=True)
            return {"verdict": deterministic, "calibration": trace}

    # (b) 텍스트 우세 경로, 또는 (a)에서 규칙이 유보한 경우
    llm_verdict, rationale = _llm_verdict(state)
    verdict, rule = psi_guard(llm_verdict, state, theta)
    trace.update(rule=rule, deterministic=False, llm_verdict=llm_verdict, llm_rationale=rationale)

    return {"verdict": verdict, "calibration": trace}


def calibrator_node(state: PrismState) -> PrismState:
    """LangGraph 노드 래퍼."""
    result = calibrate(state)
    logger.info(
        "Calibrated: %s (rule=%s, path=%s)",
        result["verdict"], result["calibration"]["rule"], result["calibration"]["path"],
    )
    return result
