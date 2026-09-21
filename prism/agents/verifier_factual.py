"""Factual Verifier (f). 항상 활성인 검증기.

게이팅된 텍스트 증거 E_T 만으로 클레임의 사실관계를 판정하고,
v_f = (y_f, gamma_f, r_f) 를 돌려준다.
"""

import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from prism.agents.guidelines import NO_EVIDENCE, RESPONSE_FORMAT, VERDICT_GUIDELINES
from prism.core.llm_models import get_llm
from prism.core.state import PrismState
from prism.core.verdict import NEI, normalize_verdict

logger = logging.getLogger(__name__)

parser = JsonOutputParser()

SYSTEM_PROMPT = f"""You are the factual verifier of a fact-checking system.
Your scope is the core factual assertion of the claim, judged against the
retrieved evidence. Entity attribution, dates and locations, and image content
are handled by other verifiers; do not rule on them.

{VERDICT_GUIDELINES}

{RESPONSE_FORMAT}"""


def factual_verifier(state: PrismState) -> PrismState:
    started = time.time()
    verdicts = state.get("verdicts", {})
    metrics = state.get("metrics", {})

    evidence = state.get("text_evidence", [])
    evidence_text = "\n\n".join(evidence) if evidence else NO_EVIDENCE

    try:
        response = get_llm(task_type="verification").invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Claim: {state['claim']}\n\nEvidence:\n{evidence_text}"),
        ])
        result = normalize_verdict(parser.parse(response.content))
    except Exception as exc:
        logger.warning("Factual 검증 실패: %s", exc)
        result = {"verdict": NEI, "confidence": 0.0, "rationale": f"verifier error: {exc}"}

    verdicts["factual"] = result
    logger.info("f: %s (%.2f)", result["verdict"], result["confidence"])

    metrics["factual_verifier_latency"] = round(time.time() - started, 4)
    return {"verdicts": verdicts, "metrics": metrics}
