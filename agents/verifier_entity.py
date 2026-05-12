import json
import spacy
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from prism.core.state import PrismState
from prism.core.llm_models import get_llm

llm = get_llm(task_type="text")
parser = JsonOutputParser()

# 💡 진경님의 기획대로 spaCy 트랜스포머 모델을 로드합니다.
# 서버가 켜질 때 한 번만 로드되므로 매번 지연이 발생하지 않습니다.
try:
    nlp = spacy.load("en_core_web_trf")
except OSError:
    print("⚠️ spaCy 모델을 찾을 수 없습니다. 터미널에서 다운로드 명령어를 실행해 주세요.")
    nlp = None

def entity_verifier(state: PrismState) -> PrismState:
    """
    spaCy NER을 통해 클레임에서 인물/기관을 명시적으로 추출하고,
    이를 바탕으로 텍스트 증거와 대조하여 귀인(Attribution)을 검증합니다.
    """
    print("--- 👤 Running Entity Verifier (with spaCy) ---")
    
    claim = state["claim"]
    evidence = "\n".join(state.get("text_evidence", []))
    
    # 1. spaCy를 이용한 개체명 인식 (NER)
    extracted_entities = []
    if nlp:
        doc = nlp(claim)
        # PERSON(사람), ORG(기관/단체), GPE(국가/도시) 태그만 골라냅니다.
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE"]:
                if ent.text not in extracted_entities:
                    extracted_entities.append(ent.text)
                    
    # 추출된 엔티티를 하나의 문자열로 묶습니다.
    entity_context = ", ".join(extracted_entities) if extracted_entities else "None specifically extracted"
    print(f"   [spaCy 추출 엔티티]: {entity_context}")
    
    # 2. 추출된 엔티티를 프롬프트에 직접 주입합니다.
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert entity verifier for a journalism fact-checking system.
Focus strictly on verifying the involvement, actions, or quotes of the following specific entities extracted from the claim:
[{entity_context}]

Check if the provided evidence supports or refutes their specific involvement.
Respond ONLY in valid JSON format with strict lowercase keys:
1. "verdict": exactly one of "Supported", "Refuted", or "NEI".
2. "confidence": a float between 0.0 and 1.0.
3. "rationale": a brief step-by-step explanation focusing on the extracted entities."""),
        ("user", "Claim: {claim}\n\nEvidence:\n{evidence}")
    ])
    
    chain = prompt | llm | parser
    
    try:
        # 프롬프트에 entity_context 변수를 추가로 넘겨줍니다.
        result = chain.invoke({
            "claim": claim, 
            "evidence": evidence,
            "entity_context": entity_context
        })
    except Exception as e:
        print(f"❌ Entity JSON 파싱 에러: {e}")
        result = {"verdict": "NEI", "confidence": 0.0, "rationale": f"Error: {e}"}
    
    current_verdicts = state.get("verdicts", {})
    current_verdicts["entity"] = result
    
    return {"verdicts": current_verdicts}