"""평가 지표.

판정 지표는 Acc와 macro-F1, 안전성 지표는 R->S(위험한 오탐)와 NEI 재현율이다.

    R->S     = |{y_hat = Supported | y = Refuted}| / |{y = Refuted}|
    NEI-rec  = |{y_hat = NEI | y = NEI}| / |{y = NEI}|

균형 이진 세트에서는 weighted-F1이 macro-F1과 같아지므로, 선행 연구가 보고한
weighted-F1과 직접 비교할 수 있다.

sklearn을 쓰지 않고 직접 계산한다. 의존성을 늘리지 않기 위해서이고,
라벨 집합이 세 개로 고정되어 있어 구현이 짧다.
"""

from collections import Counter
from typing import Dict, List, Sequence

from prism.core.verdict import NEI, REFUTED, SUPPORTED

LABELS = (SUPPORTED, REFUTED, NEI)

# 벤치마크별 라벨 표기를 내부 표기로 맞춘다.
LABEL_ALIASES = {
    "supported": SUPPORTED, "support": SUPPORTED, "true": SUPPORTED,
    "refuted": REFUTED, "refute": REFUTED, "false": REFUTED,
    "nei": NEI, "unproven": NEI, "not enough info": NEI,
}


def canonical(label: str) -> str:
    """데이터셋 라벨을 Supported / Refuted / NEI 로 정규화한다."""
    return LABEL_ALIASES.get(str(label).strip().lower(), NEI)


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def per_class_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, float]:
    """정답에 등장하는 클래스에 대해서만 F1을 계산한다."""
    present = [label for label in LABELS if label in set(y_true)]
    scores = {}
    for label in present:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        scores[label] = _f1(tp, fp, fn)
    return scores


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    scores = per_class_f1(y_true, y_pred)
    return sum(scores.values()) / len(scores) if scores else 0.0


def refuted_to_supported(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """위험한 오탐 비율. 거짓인 주장을 사실로 판정한 비율이다."""
    refuted = [p for t, p in zip(y_true, y_pred) if t == REFUTED]
    if not refuted:
        return 0.0
    return sum(1 for p in refuted if p == SUPPORTED) / len(refuted)


def nei_recall(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """기권 재현율. 증거가 불충분한 사례를 실제로 기권했는지를 본다."""
    nei = [p for t, p in zip(y_true, y_pred) if t == NEI]
    if not nei:
        return 0.0
    return sum(1 for p in nei if p == NEI) / len(nei)


def evaluate(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, object]:
    """지표를 한 번에 계산한다. 입력 라벨은 미리 정규화해 둔다."""
    true = [canonical(t) for t in y_true]
    pred = [canonical(p) for p in y_pred]

    return {
        "n": len(true),
        "accuracy": accuracy(true, pred),
        "macro_f1": macro_f1(true, pred),
        "per_class_f1": per_class_f1(true, pred),
        "r_to_s": refuted_to_supported(true, pred),
        "nei_recall": nei_recall(true, pred),
        "pred_distribution": dict(Counter(pred)),
        "true_distribution": dict(Counter(true)),
    }


def format_report(scores: Dict[str, object]) -> str:
    lines = [
        f"n = {scores['n']}",
        f"Accuracy  {scores['accuracy']:.3f}",
        f"Macro-F1  {scores['macro_f1']:.3f}",
        f"R->S      {scores['r_to_s']:.1%}",
        f"NEI-rec   {scores['nei_recall']:.3f}",
    ]
    lines.append("class F1  " + "  ".join(
        f"{label}={value:.3f}" for label, value in scores["per_class_f1"].items()
    ))
    lines.append(f"predicted {scores['pred_distribution']}")
    lines.append(f"gold      {scores['true_distribution']}")
    return "\n".join(lines)


def bootstrap_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    metric: str = "accuracy",
    iterations: int = 1000,
    seed: int = 42,
) -> List[float]:
    """부트스트랩 95% 신뢰구간 [lower, upper]."""
    import random

    fn = {"accuracy": accuracy, "macro_f1": macro_f1,
          "r_to_s": refuted_to_supported, "nei_recall": nei_recall}[metric]

    true = [canonical(t) for t in y_true]
    pred = [canonical(p) for p in y_pred]
    n = len(true)
    if n == 0:
        return [0.0, 0.0]

    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        idx = [rng.randrange(n) for _ in range(n)]
        samples.append(fn([true[i] for i in idx], [pred[i] for i in idx]))
    samples.sort()
    return [samples[int(0.025 * iterations)], samples[int(0.975 * iterations) - 1]]


def mcnemar(
    y_true: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
) -> Dict[str, float]:
    """대응 표본 McNemar 검정. 두 시스템의 정오 패턴이 다른지 본다.

    연속성 보정을 적용한 카이제곱 근사를 쓴다.
    """
    import math

    true = [canonical(t) for t in y_true]
    a = [canonical(p) for p in pred_a]
    b = [canonical(p) for p in pred_b]

    b01 = sum(1 for t, x, y in zip(true, a, b) if x == t and y != t)
    b10 = sum(1 for t, x, y in zip(true, a, b) if x != t and y == t)

    if b01 + b10 == 0:
        return {"b01": 0, "b10": 0, "statistic": 0.0, "p_value": 1.0}

    statistic = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p_value = math.erfc(math.sqrt(statistic / 2))

    return {"b01": b01, "b10": b10, "statistic": statistic, "p_value": p_value}
