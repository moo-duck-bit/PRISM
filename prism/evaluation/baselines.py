"""비교 대상 베이스라인.

B1 Majority   통계적 하한. 학습 분포의 최빈 클래스를 항상 예측한다.
B2 Zero-shot  외부 증거 없이 GPT-4o-mini로 판정한다. 파라미터 지식만의 상한을 잰다.

B3 DEFAME와 B4 MiniCheck는 외부 시스템이라 이 저장소에서 구현하지 않는다.
동일 조건 비교를 위해 두 시스템 모두 백본을 GPT-4o-mini(temperature 0)로 통제하고,
DEFAME의 오픈 웹 검색은 PRISM과 같은 증거 풀로 교체해 평가했다.
predict_from_file 로 그 결과 파일을 읽어 같은 지표 계산에 태운다.
"""

import csv
import json
import logging
from collections import Counter
from typing import Dict, List, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from prism.core.llm_models import get_llm
from prism.core.verdict import NEI, REFUTED, SUPPORTED
from prism.evaluation.dataset import Sample
from prism.evaluation.metrics import canonical

logger = logging.getLogger(__name__)

ZERO_SHOT_PROMPT = """You are a political fact-checker. Decide whether the claim is
true, false, or undecidable from what you know. You have no external evidence.

"NEI" means the claim cannot be decided without evidence you do not have.

Respond ONLY with valid JSON:
{"verdict": "Supported" | "Refuted" | "NEI"}"""


def majority(samples: Sequence[Sample]) -> List[str]:
    """B1. 최빈 클래스를 전부에 예측한다."""
    counts = Counter(s.label for s in samples)
    label = counts.most_common(1)[0][0] if counts else NEI
    logger.info("B1 majority class = %s", label)
    return [label] * len(samples)


def zero_shot(samples: Sequence[Sample]) -> List[str]:
    """B2. 증거 없이 백본 모델에게 직접 묻는다."""
    mapping = {"supported": SUPPORTED, "refuted": REFUTED, "nei": NEI}
    llm = get_llm(task_type="baseline")

    predictions = []
    for i, sample in enumerate(samples, start=1):
        user = f"Claim: {sample.claim}"
        if sample.post_text:
            user += f"\nPost: {sample.post_text}"

        try:
            response = llm.invoke([
                SystemMessage(content=ZERO_SHOT_PROMPT),
                HumanMessage(content=user),
            ])
            payload = json.loads(
                response.content.replace("```json", "").replace("```", "").strip()
            )
            raw = str(payload.get("verdict", "")).strip().lower()
            predictions.append(mapping.get(raw, NEI))
        except Exception as exc:
            logger.warning("[%d] zero-shot 실패: %s", i, exc)
            predictions.append(NEI)

    return predictions


def predict_from_file(path: str, samples: Sequence[Sample]) -> List[str]:
    """외부 시스템의 예측 파일을 읽어 샘플 순서에 맞춰 정렬한다.

    파일은 claim_id, prediction 두 컬럼을 가진 CSV여야 한다.
    누락된 claim_id는 NEI로 채우고 개수를 로그에 남긴다.
    """
    lookup: Dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lookup[str(row["claim_id"]).strip()] = canonical(row["prediction"])

    missing = 0
    predictions = []
    for sample in samples:
        if sample.claim_id in lookup:
            predictions.append(lookup[sample.claim_id])
        else:
            predictions.append(NEI)
            missing += 1

    if missing:
        logger.warning("%s 에 없는 claim_id %d건을 NEI로 채웠습니다", path, missing)
    return predictions


BASELINES = {
    "majority": majority,
    "zero_shot": zero_shot,
}
