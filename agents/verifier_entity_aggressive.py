"""
verifier_entity_aggressive.py — Anti-NEI 강화 버전

[기존 verifier_entity.py 와의 차이점]
- System Prompt에 [엄격한 판정 가이드라인] 주입
- NEI는 증거가 엔티티 주제와 '완전히 무관'할 때만 허용
- 문맥적 논리 추론이 가능하면 반드시 Supported/Refuted 중 선택

비교 실험용: 기존 파일(verifier_entity.py)은 삭제하지 않음.
"""
import spacy
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from prism.core.state import PrismState
from prism.core.llm_models import get_llm

llm = get_llm(task_type="text")
parser = JsonOutputParser()

try:
    nlp = spacy.load("en_core_web_trf")
except OSError:
    print("⚠️ spaCy 모델을 찾을 수 없습니다.")
    nlp = None

# ─── Anti-NEI 강화 시스템 프롬프트 ────────────────────────────────────────────
_SYSTEM_TEMPLATE_AGGRESSIVE = """You are an expert entity verifier for a journalism fact-checking system.
Focus strictly on verifying the involvement, actions, or quotes of the following specific entities extracted from the claim:
[{entity_context}]

Check if the provided evidence supports or refutes their specific involvement.

╔══════════════════════════════════════════════════════════════════╗
║              STRICT VERDICT GUIDELINES (MANDATORY)              ║
╠══════════════════════════════════════════════════════════════════╣
║ 1. NEI IS STRICTLY FORBIDDEN unless the evidence is COMPLETELY   ║
║    IRRELEVANT to the named entities or the claim's topic.        ║
║    NEI is the absolute LAST RESORT.                              ║
║                                                                  ║
║ 2. LOGICAL DEDUCTION REQUIRED: Use contextual reasoning even if  ║
║    the evidence doesn't explicitly mention every entity detail.  ║
║    Indirect or partial evidence is SUFFICIENT for a verdict.     ║
║                                                                  ║
║ 3. DIRECTIONAL RULE:                                             ║
║    - Evidence overall consistent with the claim → "Supported"   ║
║    - Any contradicting data about the entities → "Refuted"      ║
║                                                                  ║
║ 4. WHEN IN DOUBT: Always choose "Supported" or "Refuted".        ║
║    NEVER default to NEI out of uncertainty.                      ║
╚══════════════════════════════════════════════════════════════════╝

Respond ONLY in valid JSON format with strict lowercase keys:
1. "verdict": exactly one of "Supported", "Refuted", or "NEI" (ONLY if completely irrelevant).
2. "confidence": a float between 0.0 and 1.0.
3. "rationale": a brief step-by-step explanation focusing on the extracted entities."""


def entity_verifier_aggressive(state: PrismState) -> PrismState:
    """
    [Aggressive 버전] Anti-NEI 강화 Entity Verifier.
    spaCy NER 추출 후 Anti-NEI 지침이 주입된 프롬프트로 판정합니다.
    """
    print("--- 👤 [AGGRESSIVE] Running Entity Verifier (with spaCy) ---")

    claim = state["claim"]
    evidence = "\n".join(state.get("text_evidence", []))

    extracted_entities = []
    if nlp:
        doc = nlp(claim)
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE"]:
                if ent.text not in extracted_entities:
                    extracted_entities.append(ent.text)

    entity_context = ", ".join(extracted_entities) if extracted_entities else "None specifically extracted"
    print(f"   [spaCy 추출 엔티티]: {entity_context}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_TEMPLATE_AGGRESSIVE),
        ("user", "Claim: {claim}\n\nEvidence:\n{evidence}")
    ])

    chain = prompt | llm | parser

    try:
        result = chain.invoke({
            "claim": claim,
            "evidence": evidence if evidence.strip() else "(No external evidence retrieved. Evaluate based on the claim's internal logic.)",
            "entity_context": entity_context
        })
    except Exception as e:
        print(f"❌ [Aggressive] Entity JSON 파싱 에러: {e}")
        result = {"verdict": "NEI", "confidence": 0.0, "rationale": f"Error: {e}"}

    current_verdicts = state.get("verdicts", {})
    current_verdicts["entity"] = result

    print(f"   ✅ [Aggressive 판정 완료] Entity Verdict: {result.get('verdict')}")

    return {"verdicts": current_verdicts}
