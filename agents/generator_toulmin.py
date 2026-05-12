from langchain_core.messages import SystemMessage, HumanMessage
from prism.core.state import PrismState
from prism.core.llm_models import get_llm

# 글쓰기 작업이므로 출력을 길게 뽑을 수 있는 모델 세팅
llm = get_llm(task_type="generation") 

def toulmin_generator_agent(state: PrismState) -> PrismState:
    print("--- 📝 Running Final Article Generator (Toulmin) ---")
    
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
2. [Data/Grounds]: Summarize the evidence found.
3. [Warrant]: Explain how the evidence connects to or disproves the claim (based on the Verifier Results).
4. [Rebuttal]: Mention any missing information or conflicting perspectives (if applicable).
5. [Final Conclusion (Qualifier)]: Give a clear, final verdict (e.g., Mostly True, False, Unverified) with a brief wrap-up.

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
    return {"final_article": final_article}