"""spaCy 개체명 추출.

entity 검증기와 spatiotemporal 검증기가 같은 모델을 쓰므로 여기서 한 번만 로드한다.
spacy 임포트 자체를 함수 안으로 미룬 이유는, 모델이나 패키지가 없는 환경에서도
파이프라인 모듈을 임포트하고 테스트할 수 있어야 하기 때문이다.
모델이 없으면 NER 없이 진행하고, 검증기는 LLM 판단만으로 판정한다.

설치: python -m spacy download en_core_web_trf
"""

import logging
from functools import lru_cache
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)

MODEL_NAME = "en_core_web_trf"

PERSON_LABELS = ("PERSON", "ORG", "GPE")
SPATIOTEMPORAL_LABELS = ("DATE", "TIME", "LOC", "FAC", "GPE")

NOTHING_EXTRACTED = "None specifically extracted"


@lru_cache(maxsize=1)
def get_nlp() -> Optional[Any]:
    try:
        import spacy
    except ImportError:
        logger.warning("spacy가 설치되어 있지 않아 NER 없이 진행합니다")
        return None

    try:
        return spacy.load(MODEL_NAME)
    except OSError:
        logger.warning(
            "spaCy 모델 %s 를 찾을 수 없어 NER 없이 진행합니다. "
            "python -m spacy download %s", MODEL_NAME, MODEL_NAME,
        )
        return None


def extract(text: str, labels: Sequence[str], with_label: bool = False) -> List[str]:
    """지정한 라벨의 개체명을 등장 순서대로, 중복 없이 돌려준다.

    with_label=True면 "Rifle (GPE)" 형태로 라벨을 덧붙인다.
    시공간 검증에서는 값만 보면 지명인지 조직명인지 구분되지 않아 라벨이 필요하다.
    """
    nlp = get_nlp()
    if nlp is None:
        return []

    found: List[str] = []
    for ent in nlp(text).ents:
        if ent.label_ not in labels:
            continue
        value = f"{ent.text} ({ent.label_})" if with_label else ent.text
        if value not in found:
            found.append(value)
    return found


def as_context(entities: List[str]) -> str:
    """프롬프트에 끼워 넣을 문자열로 변환한다."""
    return ", ".join(entities) if entities else NOTHING_EXTRACTED
