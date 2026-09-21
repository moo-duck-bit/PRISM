"""태스크별 LLM 팩토리.

에이전트가 ChatOpenAI를 직접 생성하면 모델과 디코딩 설정이 파일마다 흩어져
비용도 재현성도 추적할 수 없다. 모든 호출은 이 테이블 하나를 거친다.

판정 경로(triage, verification, vision, calibration)와 심판은 전부 temperature 0이다.
따라서 같은 입력에 대해 판정 y_hat은 결정적으로 재현된다.
temperature 0.3은 판정과 분리된 설명 생성기에만 적용된다.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


@dataclass(frozen=True)
class ModelSpec:
    model: str
    temperature: float
    max_tokens: int


TASK_SPECS: Dict[str, ModelSpec] = {
    # --- 판정 경로: 전부 temperature 0 ---
    "triage":       ModelSpec("gpt-4o-mini", 0.0, 150),
    "verification": ModelSpec("gpt-4o-mini", 0.0, 200),
    "vision":       ModelSpec("gpt-4o",      0.0, 200),
    "calibration":  ModelSpec("gpt-4o-mini", 0.0, 200),

    # --- 판정과 분리된 설명 생성 ---
    "explanation":  ModelSpec("gpt-4o",      0.3, 1200),

    # --- 자동 평가 ---
    "judge":        ModelSpec("gpt-4o",      0.0, 500),
    "baseline":     ModelSpec("gpt-4o-mini", 0.0, 200),
}

# 로컬 vLLM 서버로 붙일 때 쓰는 모델.
VLLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


@lru_cache(maxsize=None)
def get_llm(task_type: str = "verification", provider: str = "openai") -> ChatOpenAI:
    """태스크에 맞는 LLM 클라이언트를 반환한다.

    태스크별로 클라이언트를 한 번만 만들어 재사용한다. 캐싱 덕분에
    임포트 시점이 아니라 첫 호출 시점에 자격 증명을 요구한다.

    Raises:
        ValueError: 정의되지 않은 task_type 또는 provider.
    """
    try:
        spec = TASK_SPECS[task_type]
    except KeyError:
        raise ValueError(
            f"정의되지 않은 task_type: {task_type!r}. 사용 가능: {', '.join(TASK_SPECS)}"
        ) from None

    if provider == "openai":
        return ChatOpenAI(
            model=spec.model,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
        )

    if provider == "vllm":
        return ChatOpenAI(
            model=VLLM_MODEL,
            openai_api_base=os.getenv("VLLM_API_BASE", "http://localhost:8000/v1"),
            openai_api_key="EMPTY",
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
        )

    raise ValueError(f"지원하지 않는 provider: {provider!r}")
