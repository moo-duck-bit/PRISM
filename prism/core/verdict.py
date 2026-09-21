"""판정값의 정규화와 집계 보조.

NEI는 버려지는 기권이 아니라 "증거 불충분"을 뜻하는 1급 판정이다.
따라서 집계에서 NEI를 임의로 제거하지 않고, 기권 여부는 캘리브레이터가 결정한다.
"""

from collections import Counter
from typing import Any, Dict, Iterable, Optional

SUPPORTED = "Supported"
REFUTED = "Refuted"
NEI = "NEI"

VALID_VERDICTS = (SUPPORTED, REFUTED, NEI)

# 텍스트 경로 검증기. 시각 검증기(v)는 교차모달 규칙에서 따로 다룬다.
TEXT_VERIFIERS = ("factual", "entity", "spatiotemporal")
VISUAL_VERIFIER = "visual"

_ALIASES = {v.lower(): v for v in VALID_VERDICTS}
_ALIASES.update({
    "support": SUPPORTED,
    "supports": SUPPORTED,
    "true": SUPPORTED,
    "refute": REFUTED,
    "refutes": REFUTED,
    "false": REFUTED,
    "not enough info": NEI,
    "not enough information": NEI,
    "unproven": NEI,
    "abstain": NEI,
})


def normalize_verdict(result: Dict[str, Any]) -> Dict[str, Any]:
    """검증기 응답을 v_s = (y_s, gamma_s, r_s) 형태로 맞춘다.

    허용 목록 밖의 라벨이나 숫자가 아닌 confidence는 각각 NEI, 0.0으로 떨어뜨린다.
    LLM 응답 표기가 흔들려도 캘리브레이션 단계에서 값이 섞이지 않게 하기 위한 장치다.
    """
    raw = str(result.get("verdict", "")).strip().lower()
    verdict = _ALIASES.get(raw, NEI)

    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "verdict": verdict,
        "confidence": min(max(confidence, 0.0), 1.0),
        "rationale": str(result.get("rationale") or ""),
    }


def vote_counts(verdicts: Dict[str, Dict[str, Any]], names: Optional[Iterable[str]] = None) -> Counter:
    """검증기 판정을 라벨별로 센다. names를 주면 해당 검증기만 대상으로 한다."""
    selected = verdicts if names is None else {k: verdicts[k] for k in names if k in verdicts}
    return Counter(
        v.get("verdict", NEI)
        for v in selected.values()
        if isinstance(v, dict)
    )


def quorum_label(counts: Counter, tau_q: int) -> Optional[str]:
    """정족수 tau_q 이상을 받은 라벨. 없으면 None."""
    for label, count in counts.most_common():
        if count >= tau_q:
            return label
    return None


def majority_label(counts: Counter) -> Optional[str]:
    """과반을 넘긴 라벨. 과반이 없거나 동수면 None."""
    total = sum(counts.values())
    if total == 0:
        return None
    label, count = counts.most_common(1)[0]
    if count * 2 > total:
        return label
    return None


def plurality_label(counts: Counter, tie_breaker: str = NEI) -> str:
    """최다 득표 라벨. 동수면 tie_breaker를 돌려준다.

    캘리브레이터를 제거한 ablation(단순 다수결)에서 쓴다.
    """
    if not counts:
        return NEI
    top_label, top_count = counts.most_common(1)[0]
    if list(counts.values()).count(top_count) > 1:
        return tie_breaker
    return top_label
