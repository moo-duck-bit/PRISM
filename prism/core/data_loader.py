"""MOCHEG 데이터셋 로더."""

import logging
from typing import Any, Dict, List

from datasets import load_dataset

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "vcmt794/mocheg-train"


def load_mocheg(repo_id: str = DEFAULT_REPO_ID, split_ratio: int = 50) -> List[Dict[str, Any]]:
    """허깅페이스에서 데이터셋의 앞 split_ratio(%) 구간만 내려받는다.

    전체를 받으면 개발 반복이 느려지므로 기본값은 절반이다.
    로드에 실패하면 예외를 올리지 않고 빈 리스트를 반환한다.
    """
    split = f"train[:{split_ratio}%]"
    logger.info("%s 로드 중 (%s)", repo_id, split)

    try:
        dataset = load_dataset(repo_id, split=split)
    except Exception as exc:
        logger.error("데이터 로드 실패: %s", exc)
        return []

    logger.info("샘플 %d개 로드 완료", len(dataset))
    return list(dataset)


def preview(samples: List[Dict[str, Any]], max_chars: int = 100) -> None:
    """첫 샘플의 키와 값을 잘라서 출력한다. 스키마 확인용."""
    if not samples:
        logger.warning("미리 볼 샘플이 없습니다")
        return

    for key, value in samples[0].items():
        text = str(value)
        suffix = "..." if len(text) > max_chars else ""
        print(f"- {key}: {text[:max_chars]}{suffix}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    preview(load_mocheg(split_ratio=50))
