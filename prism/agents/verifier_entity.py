"""Entity Verifier (e). triage 마스크 m_ent 가 켜졌을 때만 활성화된다.

클레임에 등장한 인물·기관에 어떤 발언이나 행위가 귀속되는지를 검증한다.
클레임 전체를 넘기면 관할이 흐려지므로, spaCy로 대상을 먼저 뽑아 프롬프트에 못박는다.
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
    ("system", f"""You are the entity verifier of a fact-checking system.
Your scope is limited to these entities extracted from the claim:
[{{entity_context}}]

Verify whether the evidence supports or refutes their involvement, actions, or
quoted statements. Do not rule on dates, locations, or image content.

{VERDICT_GUIDELINES}

{RESPONSE_FORMAT}"""),
    ("user", "Claim: {claim}\n\nEvidence:\n{evidence}"),
])


def entity_verifier(state: PrismState) -> PrismState:
    started = time.time()
    claim = state["claim"]

    entity_context = ner.as_context(ner.extract(claim, ner.PERSON_LABELS))
    evidence = "\n".join(state.get("text_evidence", [])) or NO_EVIDENCE
    logger.info("e: 추출 개체명 %s", entity_context)

    try:
        chain = PROMPT | get_llm(task_type="verification") | parser
        result = normalize_verdict(chain.invoke({
            "claim": claim,
            "evidence": evidence,
            "entity_context": entity_context,
        }))
    except Exception as exc:
        logger.warning("Entity 검증 실패: %s", exc)
        result = {"verdict": NEI, "confidence": 0.0, "rationale": f"verifier error: {exc}"}

    verdicts = state.get("verdicts", {})
    verdicts["entity"] = result
    logger.info("e: %s (%.2f)", result["verdict"], result["confidence"])

    metrics = state.get("metrics", {})
    metrics["entity_verifier_latency"] = round(time.time() - started, 4)
    return {"verdicts": verdicts, "metrics": metrics}
