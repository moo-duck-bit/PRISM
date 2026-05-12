import os
import base64
import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from prism.core.state import PrismState

# 진경님이 설정하신 GPT-4o 모델
vision_llm = ChatOpenAI(model="gpt-4o", temperature=0)

def encode_image(image_path: str) -> str:
    """💡 핵심 해결책: 로컬 이미지를 읽어 Base64 텍스트 문자열로 변환합니다."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def visual_verifier(state: PrismState) -> PrismState:
    print("--- 👁️ Running Visual Verifier ---")
    
    claim = state.get("claim", "This image contains specific identifiable entities or events.")
    image_paths = state.get("image_evidence", [])
    current_verdicts = state.get("verdicts", {})

    if not image_paths or not os.path.exists(image_paths[0]):
        print("   [경고] 유효한 이미지 증거가 없습니다.")
        current_verdicts["visual"] = {"verdict": "NEI", "confidence": 0.0, "rationale": "No valid image provided."}
        return {"verdicts": current_verdicts}

    target_image_path = image_paths[0]
    
    # 이미지를 텍스트(Base64)로 변환
    base64_image = encode_image(target_image_path)

    system_prompt = """You are an expert Visual Image Forensics and Fact-Checker.
You will be provided with a Claim and an Image.
Respond ONLY in valid JSON format:
{"verdict": "Supported" or "Refuted" or "NEI", "confidence": 0.0, "rationale": "short explanation"}"""

    # 💡 LangChain에 맞춘 멀티모달 메시지 포맷 (Base64 데이터 주입)
    user_message = HumanMessage(
        content=[
            {"type": "text", "text": f"Claim: {claim}\nAnalyze the attached image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    )

    print(f"   ⏳ GPT-4o가 이미지({os.path.basename(target_image_path)})를 텍스트로 변환해 눈으로 분석 중입니다...")
    try:
        response = vision_llm.invoke([SystemMessage(content=system_prompt), user_message])
        
        # JSON 파싱 (안전 장치 추가)
        clean_content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_content)
        
    except Exception as e:
        print(f"❌ Visual Verifier 파싱 에러: {e}")
        result = {"verdict": "NEI", "confidence": 0.0, "rationale": f"Error: {e}"}

    current_verdicts["visual"] = result
    print(f"   ✅ [시각 판정 완료] {result.get('verdict')}")
    
    return {"verdicts": current_verdicts}