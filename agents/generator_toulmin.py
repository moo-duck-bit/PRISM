import time
from langchain_core.messages import SystemMessage, HumanMessage
from prism.core.state import PrismState
from prism.core.llm_models import get_llm

# 글쓰기 작업이므로 출력을 길게 뽑을 수 있는 모델 세팅
llm = get_llm(task_type="generation") 

def toulmin_generator_agent(state: PrismState) -> PrismState:
    print("--- 📝 Running Final Article Generator (Toulmin) ---")
    _start = time.time()

    claim = state.get("claim", "")
    verdicts = state.get("verdicts", {})
    
    # 각 검증기들의 판정 결과를 문자열로 예쁘게 묶습니다.
    verdict_summary = ""
    for v_type, result in verdicts.items():
        verdict_summary += f"- {v_type.capitalize()} Check: {result.get('verdict')} (Confidence: {result.get('confidence')})\n  Reason: {result.get('rationale')}\n"

    system_prompt = """You are an expert investigative journalist and fact-checker.
Write a comprehensive fact-checking report based on the provided Claim and the Verifier Results.
Structure your article strictly using the Toulmin Model of Argumentation:

1. [Claim]: State the original claim being checked.
2. [Data/Grounds]: Summarize ONLY the evidence explicitly present in the Verifier Results.
3. [Warrant]: Explain how the evidence logically connects to or disproves the claim.
4. [Rebuttal]: Mention any conflicting perspectives found in the evidence (if applicable).
5. [Final Conclusion (Qualifier)]: Give a clear, final verdict (e.g., Mostly True, False, Unverified) with a brief wrap-up.

╔══════════════════════════════════════════════════════════════════╗
║           HALLUCINATION PREVENTION RULES (MANDATORY)            ║
╠══════════════════════════════════════════════════════════════════╣
║ 1. EVIDENCE-ONLY RULE: NEVER invent, fabricate, or infer facts   ║
║    that are not explicitly stated in the provided Verifier       ║
║    Results or Evidence text. Every factual claim in your report  ║
║    must be traceable to the provided evidence.                   ║
║                                                                  ║
║ 2. WARRANT SAFETY RULE: If the logical connection (Warrant) or   ║
║    Backing is NOT explicitly stated in the evidence and is only  ║
║    implicitly assumed, DO NOT fabricate an elaborate explanation. ║
║    Instead, write briefly:                                       ║
║    "The warrant is implicitly supported by the evidence          ║
║     (증거에 의해 암묵적으로 지지됨)."                                ║
║                                                                  ║
║ 3. UNCERTAINTY RULE: If a piece of information is uncertain,     ║
║    clearly mark it as such (e.g., "according to the evidence",   ║
║    "증거에 따르면"). Never present uncertain info as established   ║
║    fact.                                                         ║
╚══════════════════════════════════════════════════════════════════╝

Write the report in clear, professional Korean."""

    user_message = f"Claim to check: {claim}\n\nVerifier Results:\n{verdict_summary}"
    
    print("   ⏳ 팩트체킹 보고서를 작성하고 있습니다...")
    try:
        ai_msg = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ])
        final_article = ai_msg.content
    except Exception as e:
        print(f"❌ 기사 작성 에러: {e}")
        final_article = "보고서 작성에 실패했습니다."

    print("   ✅ [기사 작성 완료]")

    metrics = state.get("metrics", {})
    metrics["generator_latency"] = round(time.time() - _start, 4)

    return {"final_article": final_article, "metrics": metrics}