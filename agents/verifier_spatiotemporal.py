import json
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
    print("⚠️ spaCy 모델을 찾을 수 없습니다. 터미널에서 다운로드 명령어를 실행해 주세요.")
    nlp = None

def spatiotemporal_verifier(state: PrismState) -> PrismState:
    """
    spaCy NER을 통해 클레임에서 시간/장소를 추출하고,
    이를 바탕으로 텍스트 증거와 대조하여 시공간적 일치 여부를 검증합니다.
    """
    print("--- ⏱️ Running Spatiotemporal Verifier (with spaCy) ---")
    
    claim = state["claim"]
    evidence = "\n".join(state.get("text_evidence", []))
    
    # 1. spaCy를 이용한 개체명 인식 (NER - 시간/장소 집중)
    extracted_elements = []
    if nlp:
        doc = nlp(claim)
        # DATE(날짜), TIME(시간), LOC(위치), FAC(시설물), GPE(국가/도시) 태그만 골라냅니다.
        for ent in doc.ents:
            if ent.label_ in ["DATE", "TIME", "LOC", "FAC", "GPE"]:
                if ent.text not in extracted_elements:
                    # 확인하기 쉽도록 단어 뒤에 태그(예: Rifle(GPE))를 붙여줍니다.
                    extracted_elements.append(f"{ent.text} ({ent.label_})")
                    
    spatiotemporal_context = ", ".join(extracted_elements) if extracted_elements else "None specifically extracted"
    print(f"   [spaCy 추출 시간/장소]: {spatiotemporal_context}")
    
    # 2. 추출된 시간/장소를 프롬프트에 직접 주입합니다.
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert spatiotemporal verifier for a journalism fact-checking system.
Focus strictly on verifying the specific time, date, and location elements extracted from the claim:
[{spatiotemporal_context}]

Check if the provided evidence supports or refutes these specific spatiotemporal details.
Respond ONLY in valid JSON format with strict lowercase keys:
1. "verdict": exactly one of "Supported", "Refuted", or "NEI".
2. "confidence": a float between 0.0 and 1.0.
3. "rationale": a brief step-by-step explanation focusing ONLY on the extracted time and location."""),
        ("user", "Claim: {claim}\n\nEvidence:\n{evidence}")
    ])
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "claim": claim, 
            "evidence": evidence,
            "spatiotemporal_context": spatiotemporal_context
        })
    except Exception as e:
        print(f"❌ Spatiotemporal JSON 파싱 에러: {e}")
        result = {"verdict": "NEI", "confidence": 0.0, "rationale": f"Error parsing JSON: {e}"}
    
    # verdicts 딕셔너리에 spatiotemporal 결과를 나란히 추가합니다.
    current_verdicts = state.get("verdicts", {})
    current_verdicts["spatiotemporal"] = result
    
    return {"verdicts": current_verdicts}