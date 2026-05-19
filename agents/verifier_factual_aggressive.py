"""
verifier_factual_aggressive.py — Anti-NEI 강화 버전

[기존 verifier_factual.py 와의 차이점]
1. System Prompt에 [엄격한 판정 가이드라인] 주입
   - NEI는 증거가 주제와 '완전히 무관'할 때만 허용
   - 문맥적 논리 추론이 가능하면 반드시 Supported/Refuted 중 선택
2. 증거 없음(empty evidence) 시 NEI 즉시 반환 → LLM에게 판단 위임으로 변경
   (증거가 없더라도 claim 자체에서 논리 추론 시도)

비교 실험용: 기존 파일(verifier_factual.py)은 삭제하지 않음.
"""
import time
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from prism.core.state import PrismState
from prism.core.llm_models import get_llm

llm = get_llm(task_type="verification")
parser = JsonOutputParser()

# ─── Anti-NEI 강화 시스템 프롬프트 ────────────────────────────────────────────
_SYSTEM_PROMPT_AGGRESSIVE = """You are an expert factual verifier for a fact-checking system.
You will be given a 'Claim' and a list of 'Evidence' documents.
Determine if the evidence supports or refutes the claim.

╔══════════════════════════════════════════════════════════════════╗
║              STRICT VERDICT GUIDELINES (MANDATORY)              ║
╠══════════════════════════════════════════════════════════════════╣
║ 1. NEI IS STRICTLY FORBIDDEN unless the evidence is COMPLETELY   ║
║    IRRELEVANT to the claim's topic (e.g., evidence about an      ║
║    unrelated subject). NEI is the absolute LAST RESORT.          ║
║                                                                  ║
║ 2. LOGICAL DEDUCTION REQUIRED: Even if the evidence does NOT     ║
║    match the claim word-for-word, you MUST use contextual        ║
║    reasoning and logical deduction to reach a verdict.           ║
║    Partial or indirect evidence is SUFFICIENT for a verdict.     ║
║                                                                  ║
║ 3. DIRECTIONAL RULE:                                             ║
║    - If evidence OVERALL supports the claim's assertion          ║
║      → verdict = "Supported"                                     ║
║    - If even ONE piece of contradicting or inconsistent data     ║
║      exists in the evidence → verdict = "Refuted"               ║
║                                                                  ║
║ 4. WHEN IN DOUBT: Always choose "Supported" or "Refuted".        ║
║    NEVER default to NEI out of uncertainty.                      ║
╚══════════════════════════════════════════════════════════════════╝

Respond ONLY in valid JSON format with strict lowercase keys:
1. "verdict": exactly one of "Supported", "Refuted", or "NEI" (ONLY if completely irrelevant).
2. "confidence": a float between 0.0 and 1.0.
3. "rationale": a short, logical explanation of your decision based on the provided evidence."""


def factual_verifier_aggressive(state: PrismState) -> PrismState:
    """
    [Aggressive 버전] Anti-NEI 강화 Factual Verifier.
    증거가 없어도 NEI를 즉시 반환하지 않고 LLM에게 판단을 위임합니다.
    """
    print("--- 🧠 [AGGRESSIVE] Running Factual Verifier ---")
    _start = time.time()

    triage_results = state.get("triage_results", {})
    if not triage_results.get("factual_needed", True):
        print("   [건너뜀] Triage 결과 Factual 검증이 필요하지 않습니다.")
        metrics = state.get("metrics", {})
        metrics["factual_verifier_latency"] = round(time.time() - _start, 4)
        return {"verdicts": state.get("verdicts", {}), "metrics": metrics}

    claim = state["claim"]
    evidence_list = state.get("text_evidence", [])

    # [Aggressive 변경점] 증거 없음 → 즉시 NEI 반환 대신, "No external evidence found"를 알리고 LLM 판단 위임
    if not evidence_list:
        print("   [경고] 텍스트 증거가 없음 → LLM에게 claim 자체로 판단 위임 (Aggressive Mode)")
        evidence_text = "(No external evidence retrieved. Evaluate based on the claim's internal logic and general knowledge.)"
    else:
        evidence_text = "\n\n".join(evidence_list)

    user_message = f"Claim: {claim}\n\nEvidence:\n{evidence_text}"

    print("   ⏳ [Aggressive] GPT가 반드시 Supported/Refuted 판정을 내립니다...")
    try:
        ai_msg = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT_AGGRESSIVE),
            HumanMessage(content=user_message)
        ])
        result = parser.parse(ai_msg.content)
    except Exception as e:
        print(f"❌ [Aggressive] Factual Verifier 파싱 에러: {e}")
        result = {"verdict": "NEI", "confidence": 0.0, "rationale": f"Error: {e}"}

    current_verdicts = state.get("verdicts", {})
    current_verdicts["factual"] = result

    print(f"   ✅ [Aggressive 판정 완료] Verdict: {result.get('verdict')} (확신도: {result.get('confidence')})")

    metrics = state.get("metrics", {})
    metrics["factual_verifier_latency"] = round(time.time() - _start, 4)

    return {"verdicts": current_verdicts, "metrics": metrics}
