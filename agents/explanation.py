import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from prism.core.state import PrismState
from prism.core.llm_models import get_llm

# 설명을 작성해야 하므로 max_tokens를 넉넉히 쓸 수 있도록 설정하거나 기본값을 씁니다.
# 툴민 구조는 텍스트가 조금 길어질 수 있으므로 토큰 제한을 풀어준 별도의 llm을 쓸 수도 있지만,
# 우선 우리가 만든 기본 텍스트 팩토리 모델을 사용합니다.
llm = get_llm(task_type="text") 
parser = JsonOutputParser()

def explanation_agent(state: PrismState) -> PrismState:
    """
    수집된 증거와 판정 결과를 바탕으로 Toulmin 논증 구조에 맞춰 최종 설명을 생성합니다.
    """
    print("--- ✍️ Running Explanation Agent (Toulmin Structure) ---")
    
    claim = state["claim"]
    evidence = "\n".join(state.get("text_evidence", []))
    verdicts = json.dumps(state.get("verdicts", {}), ensure_ascii=False)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert journalist writing a final fact-checking report.
Based on the Claim, Evidence, and the Agents' Verdicts, generate a final explanation using the Toulmin model of argumentation.
Respond ONLY in valid JSON format with strict lowercase keys:
1. "claim_statement": A clear sentence stating if the original claim is Supported, Refuted, or NEI.
2. "grounds": A brief summary of the actual evidence found.
3. "warrant": An explanation of WHY the evidence connects to and proves/disproves the claim.
4. "backing": The sources or context of the evidence."""),
        ("user", "Claim: {claim}\n\nEvidence:\n{evidence}\n\nVerdicts:\n{verdicts}")
    ])
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({"claim": claim, "evidence": evidence, "verdicts": verdicts})
    except Exception as e:
        print(f"❌ Explanation JSON 파싱 에러: {e}")
        result = {"error": "Failed to generate explanation."}
        
    # 상태(State)에 'explanation' 이라는 새로운 공간을 만들어 저장합니다.
    return {"explanation": result}