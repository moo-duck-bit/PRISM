"""Visual Verifier (v). m_img 가 켜져 있고 실제 이미지가 있을 때만 활성화된다.

이 검증기의 confidence gamma_v 는 교차모달 충돌 규칙에서 임계값 tau_v 와 직접
비교되므로, 프롬프트가 confidence를 근거 강도에 연동하도록 명시한다.

LangChain 멀티모달 메시지는 URL 또는 data URI만 받으므로 로컬 파일은 base64로 넘긴다.
"""

import base64
import json
import logging
import time
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from prism.agents.guidelines import RESPONSE_FORMAT, VERDICT_GUIDELINES
from prism.core.llm_models import get_llm
from prism.core.state import PrismState
from prism.core.verdict import NEI, normalize_verdict

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = f"""You are the visual verifier of a fact-checking system.
Your scope is whether the image is consistent with the claim: what it actually
depicts, when and where it plausibly comes from, and whether it has been used
out of context. Do not rule on parts of the claim the image cannot speak to.

{VERDICT_GUIDELINES}

Set confidence to how decisively the image itself settles the question. Reserve
confidence above 0.7 for cases where the image clearly contradicts or clearly
corroborates the claim, because a high-confidence visual refutation is allowed
to override the text verifiers.

{RESPONSE_FORMAT}"""

NO_IMAGE_VERDICT: Dict[str, Any] = {
    "verdict": NEI,
    "confidence": 0.0,
    "rationale": "No image evidence was available.",
}


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_json(content: str) -> Dict[str, Any]:
    """Vision 모델은 JSON을 코드펜스로 감싸는 경우가 잦아 먼저 걷어낸다."""
    return json.loads(content.replace("```json", "").replace("```", "").strip())


def visual_verifier(state: PrismState) -> PrismState:
    started = time.time()
    verdicts = state.get("verdicts", {})
    metrics = state.get("metrics", {})
    images = state.get("image_evidence", [])

    if not images:
        # 게이팅을 통과하지 못한 경우. 시각적으로 말할 수 있는 것이 없으므로 기권한다.
        verdicts["visual"] = dict(NO_IMAGE_VERDICT)
        metrics["visual_verifier_latency"] = round(time.time() - started, 4)
        return {"verdicts": verdicts, "metrics": metrics}

    image_path = images[0]
    prompt = f"Claim: {state.get('claim', '')}"
    if state.get("ocr_text"):
        prompt += f"\nText detected in the image (OCR): {state['ocr_text']}"
    prompt += "\nAnalyze the attached image."

    try:
        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"},
            },
        ])
        response = get_llm(task_type="vision").invoke(
            [SystemMessage(content=SYSTEM_PROMPT), message]
        )
        result = normalize_verdict(parse_json(response.content))
    except Exception as exc:
        logger.warning("Visual 검증 실패: %s", exc)
        result = {"verdict": NEI, "confidence": 0.0, "rationale": f"verifier error: {exc}"}

    verdicts["visual"] = result
    logger.info("v: %s (gamma_v=%.2f)", result["verdict"], result["confidence"])

    metrics["visual_verifier_latency"] = round(time.time() - started, 4)
    return {"verdicts": verdicts, "metrics": metrics}
