import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from prism.core.state import PrismState
from prism.core.llm_models import get_llm

llm = get_llm(task_type="text")
parser = JsonOutputParser()

def triage_agent(state: PrismState) -> PrismState:
    """
    클레임을 분석하여 어떤 검증 에이전트를 활성화할지 결정하는 라우팅 에이전트입니다.
    """
    print("--- Running Modality-Aware Triage Agent ---")
    claim = state["claim"]
    
    # 💡 수정된 부분: 프롬프트에 "소문자 true/false를 엄격히 사용하라"는 지침을 추가했습니다.
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert journalism triage agent. Analyze the claim and decide the verification routing.
Respond ONLY in valid JSON format with three boolean keys (use strictly lowercase true or false) and one string key for reasoning:
1. "image_needed": true if the claim explicitly mentions a photo, video, or if visual evidence is crucial to verify it.
2. "temporal": true if the claim heavily relies on a specific time, date, or sequence of events.
3. "entity": true if the claim attributes a specific quote, action, or status to a named person or organization.
4. "reasoning": A brief 1-2 sentence explanation of why you assigned true or false to the keys above."""),
        ("user", "Claim: {claim}")
    ])
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({"claim": claim})
    except Exception as e:
        print(f"❌ JSON 파싱 에러 발생: {e}")
        result = {"image_needed": False, "temporal": False, "entity": False, "reasoning": "Error parsing JSON."}
        
    return {"triage_results": result}