"""
verifier_spatiotemporal_aggressive.py — Anti-NEI 강화 버전

[기존 verifier_spatiotemporal.py 와의 차이점]
- System Prompt에 [엄격한 판정 가이드라인] 주입
- NEI는 증거가 시간/장소 주제와 '완전히 무관'할 때만 허용
- 문맥적 논리 추론이 가능하면 반드시 Supported/Refuted 중 선택

비교 실험용: 기존 파일(verifier_spatiotemporal.py)은 삭제하지 않음.
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
_SYSTEM_TEMPLATE_AGGRESSIVE = """You are an expert spatiotemporal verifier for a journalism fact-checking system.
Focus strictly on verifying the specific time, date, and location elements extracted from the claim:
[{spatiotemporal_context}]

Check if the provided evidence supports or refutes these specific spatiotemporal details.

╔══════════════════════════════════════════════════════════════════╗
║              STRICT VERDICT GUIDELINES (MANDATORY)              ║
╠══════════════════════════════════════════════════════════════════╣
║ 1. NEI IS STRICTLY FORBIDDEN unless the evidence is COMPLETELY   ║
║    IRRELEVANT to the claim's time, date, or location context.    ║
║    NEI is the absolute LAST RESORT.                              ║
║                                                                  ║
║ 2. LOGICAL DEDUCTION REQUIRED: Even if dates/locations are only  ║
║    partially mentioned, use contextual reasoning to reach a      ║
║    verdict. Approximate or implied temporal/spatial matches are  ║
║    SUFFICIENT for a verdict.                                     ║
║                                                                  ║
║ 3. DIRECTIONAL RULE:                                             ║
║    - Evidence consistent with the claim's time/place → "Supported"║
║    - Any contradicting temporal/spatial data → "Refuted"         ║
║                                                                  ║
║ 4. WHEN IN DOUBT: Always choose "Supported" or "Refuted".        ║
║    NEVER default to NEI out of uncertainty.                      ║
╚══════════════════════════════════════════════════════════════════╝

Respond ONLY in valid JSON format with strict lowercase keys:
1. "verdict": exactly one of "Supported", "Refuted", or "NEI" (ONLY if completely irrelevant).
2. "confidence": a float between 0.0 and 1.0.
3. "rationale": a brief step-by-step explanation focusing ONLY on the extracted time and location."""


def spatiotemporal_verifier_aggressive(state: PrismState) -> PrismState:
    """
    [Aggressive 버전] Anti-NEI 강화 Spatiotemporal Verifier.
    spaCy NER 추출 후 Anti-NEI 지침이 주입된 프롬프트로 판정합니다.
    """
    print("--- ⏱️ [AGGRESSIVE] Running Spatiotemporal Verifier (with spaCy) ---")

    claim = state["claim"]
    evidence = "\n".join(state.get("text_evidence", []))

    extracted_elements = []
    if nlp:
        doc = nlp(claim)
        for ent in doc.ents:
            if ent.label_ in ["DATE", "TIME", "LOC", "FAC", "GPE"]:
                if ent.text not in extracted_elements:
                    extracted_elements.append(f"{ent.text} ({ent.label_})")

    spatiotemporal_context = ", ".join(extracted_elements) if extracted_elements else "None specifically extracted"
    print(f"   [spaCy 추출 시간/장소]: {spatiotemporal_context}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_TEMPLATE_AGGRESSIVE),
        ("user", "Claim: {claim}\n\nEvidence:\n{evidence}")
    ])

    chain = prompt | llm | parser

    try:
        result = chain.invoke({
            "claim": claim,
            "evidence": evidence if evidence.strip() else "(No external evidence retrieved. Evaluate based on the claim's internal logic.)",
            "spatiotemporal_context": spatiotemporal_context
        })
    except Exception as e:
        print(f"❌ [Aggressive] Spatiotemporal JSON 파싱 에러: {e}")
        result = {"verdict": "NEI", "confidence": 0.0, "rationale": f"Error parsing JSON: {e}"}

    current_verdicts = state.get("verdicts", {})
    current_verdicts["spatiotemporal"] = result

    print(f"   ✅ [Aggressive 판정 완료] Spatiotemporal Verdict: {result.get('verdict')}")

    return {"verdicts": current_verdicts}
