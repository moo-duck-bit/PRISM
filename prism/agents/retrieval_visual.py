"""Gated Evidence Retrieval (논문 3.3절, 시각 경로).

    E_I = I  if m_img and |I| >= 1  else  empty

입력에 딸려 온 이미지를 그대로 증거로 삼되, triage 마스크가 켜져 있고 실제
이미지가 존재할 때만 통과시킨다. 텍스트 경로와 달리 별도 검색을 하지 않는 이유는,
대상 이미지가 클레임과 함께 제시되는 포스트 단위 입력이기 때문이다.
"""

import logging
import os
import time
from typing import List

from prism.core.state import PrismState

logger = logging.getLogger(__name__)


def gate_images(image_paths: List[str], image_needed: bool) -> List[str]:
    """마스크와 파일 존재 여부를 모두 만족하는 이미지만 남긴다."""
    if not image_needed:
        return []
    return [path for path in image_paths if path and os.path.exists(path)]


def visual_retrieval_agent(state: PrismState) -> PrismState:
    started = time.time()
    metrics = state.get("metrics", {})

    images = gate_images(
        state.get("image_paths", []),
        image_needed=bool(state.get("modality_mask", {}).get("img")),
    )

    if images:
        metrics["image_retrieval_calls"] = metrics.get("image_retrieval_calls", 0) + 1
        logger.info("E_I: 이미지 %d건", len(images))
    else:
        logger.info("E_I: 비어 있음 (마스크 꺼짐 또는 유효 이미지 없음)")

    metrics["retrieval_visual_latency"] = round(time.time() - started, 4)
    return {"image_evidence": images, "metrics": metrics}
