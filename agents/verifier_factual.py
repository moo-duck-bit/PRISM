import json
import time
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from prism.core.state import PrismState
from prism.core.llm_models import get_llm

# Factual 검증을 위한 LLM 호출
llm = get_llm(task_type="verification")
parser = JsonOutputParser()

def factual_verifier(state: PrismState) -> PrismState:
    """
    검색된 텍스트 증거를 바탕으로 클레임의 사실 여부를 판별합니다.
    """
    print("--- 🧠 Running Factual Verifier ---")
    _start = time.time()

    # 1. Triage 지시 확인 (가짜 데이터 시절 만들어둔 로직 연동)
    triage_results = state.get("triage_results", {})
    if not triage_results.get("factual_needed", True):
        print("   [건너뜀] Triage 결과 Factual 검증이 필요하지 않습니다.")
        metrics = state.get("metrics", {})
        metrics["factual_verifier_latency"] = round(time.time() - _start, 4)
        return {"verdicts": state.get("verdicts", {}), "metrics": metrics}

    claim = state["claim"]
    evidence_list = state.get("text_evidence", [])
    
    # 2. 검색된 증거가 없으면 LLM에게 판단 위임 (즉시 NEI 반환 금지)
    if not evidence_list:
        print("   [경고] 텍스트 증거 없음 → LLM에게 claim 자체로 판단 위임")
        evidence_list = ["(No external evidence retrieved. Use the claim's internal logic and general knowledge to reason.)"]

    # 3. LLM에게 지시할 프롬프트 구성
    system_prompt = """You are an expert factual verifier for a fact-checking system.
You will be given a 'Claim' and a list of 'Evidence' documents.
Determine if the evidence supports or refutes the claim.

╔══════════════════════════════════════════════════════════════════╗
║              STRICT VERDICT GUIDELINES (MANDATORY)              ║
╠══════════════════════════════════════════════════════════════════╣
║ 1. NEI SUPPRESSION: Do NOT return "NEI" unless the evidence is   ║
║    COMPLETELY IRRELEVANT to the claim's topic. NEI is the        ║
║    absolute LAST RESORT.                                         ║
║    Even if evidence does not match word-for-word, use logical    ║
║    deduction (Logical Deduction) to reach a verdict.             ║
║                                                                  ║
║ 2. DIRECTIONAL HEURISTIC (apply in order):                       ║
║    a) If the overall tone/data of the evidence is CONSISTENT     ║
║       with the claim's direction → verdict = "Supported"         ║
║    b) If even ONE piece of evidence CONTRADICTS the claim's      ║
║       core assertion → verdict = "Refuted"                       ║
║    c) Only if the evidence is entirely off-topic → "NEI"         ║
║                                                                  ║
║ 3. WHEN IN DOUBT: Always choose "Supported" or "Refuted".        ║
╚══════════════════════════════════════════════════════════════════╝

Respond ONLY in valid JSON format with strict lowercase keys:
1. "verdict": exactly one of "Supported", "Refuted", or "NEI" (only if completely irrelevant).
2. "confidence": a float between 0.0 and 1.0.
3. "rationale": a short, logical explanation of your decision based ONLY on the provided evidence."""

    # 잘라둔 3개의 증거를 보기 좋게 하나의 텍스트로 합칩니다.
    evidence_text = "\n\n".join(evidence_list)
    user_message = f"Claim: {claim}\n\nEvidence:\n{evidence_text}"

    # 4. LLM 호출 및 JSON 파싱
    print("   ⏳ GPT가 증거를 읽고 진위를 판별하고 있습니다...")
    try:
        ai_msg = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ])
        result = parser.parse(ai_msg.content)
    except Exception as e:
        print(f"❌ Factual Verifier 파싱 에러: {e}")
        result = {"verdict": "NEI", "confidence": 0.0, "rationale": f"Error: {e}"}

    # 5. 판정 결과 저장
    current_verdicts = state.get("verdicts", {})
    current_verdicts["factual"] = result

    print(f"   ✅ [판정 완료] Verdict: {result.get('verdict')} (확신도: {result.get('confidence')})")

    metrics = state.get("metrics", {})
    metrics["factual_verifier_latency"] = round(time.time() - _start, 4)

    return {"verdicts": current_verdicts, "metrics": metrics}