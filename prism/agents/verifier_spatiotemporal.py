"""Spatiotemporal Verifier (st). triage 마스크 m_time 이 켜졌을 때만 활성화된다.

클레임의 날짜·시각·장소가 증거와 일치하는지만 검증한다.
"""

import logging
import time

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from prism.agents import ner
from prism.agents.guidelines import NO_EVIDENCE, RESPONSE_FORMAT, VERDICT_GUIDELINES
from prism.core.llm_models import get_llm
from prism.core.state import PrismState
from prism.core.verdict import NEI, normalize_verdict

logger = logging.getLogger(__name__)

parser = JsonOutputParser()

PROMPT = ChatPromptTemplate.from_messages([
    ("system", f"""You are the spatiotemporal verifier of a fact-checking system.
Your scope is limited to these time and place elements extracted from the claim:
[{{spatiotemporal_context}}]

Verify whether the evidence supports or refutes these specific details. Do not
rule on the claim's core assertion, entity attribution, or image content.

{VERDICT_GUIDELINES}

{RESPONSE_FORMAT}"""),
    ("user", "Claim: {claim}\n\nEvidence:\n{evidence}"),
])


def spatiotemporal_verifier(state: PrismState) -> PrismState:
    started = time.time()
    claim = state["claim"]

    # GPE는 개체명 검증과 라벨이 겹쳐, 값만 넘기면 지명인지 조직명인지 구분되지 않는다.
    elements = ner.extract(claim, ner.SPATIOTEMPORAL_LABELS, with_label=True)
    context = ner.as_context(elements)
    evidence = "\n".join(state.get("text_evidence", [])) or NO_EVIDENCE
    logger.info("st: 추출 시공간 요소 %s", context)

    try:
        chain = PROMPT | get_llm(task_type="verification") | parser
        result = normalize_verdict(chain.invoke({
            "claim": claim,
            "evidence": evidence,
            "spatiotemporal_context": context,
        }))
    except Exception as exc:
        logger.warning("Spatiotemporal 검증 실패: %s", exc)
        result = {"verdict": NEI, "confidence": 0.0, "rationale": f"verifier error: {exc}"}

    verdicts = state.get("verdicts", {})
    verdicts["spatiotemporal"] = result
    logger.info("st: %s (%.2f)", result["verdict"], result["confidence"])

    metrics = state.get("metrics", {})
    metrics["spatiotemporal_verifier_latency"] = round(time.time() - started, 4)
    return {"verdicts": verdicts, "metrics": metrics}
