"""
verifier_visual_aggressive.py — Anti-NEI 강화 버전

[기존 verifier_visual.py 와의 차이점]
- System Prompt에 [엄격한 판정 가이드라인] 주입
- 이미지 없음 시 NEI 즉시 반환 대신 텍스트 증거 기반 판단으로 폴백
  (이미지가 없는 경우 visual 검증 자체가 의미 없으므로, claim 기반으로 Supported 폴백 처리)
  → 이미지 없는 케이스에서 NEI 투표가 쌓이는 것을 방지

비교 실험용: 기존 파일(verifier_visual.py)은 삭제하지 않음.
"""
import os
import base64
import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from prism.core.state import PrismState

vision_llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ─── Anti-NEI 강화 시스템 프롬프트 ────────────────────────────────────────────
_SYSTEM_PROMPT_AGGRESSIVE = """You are an expert Visual Image Forensics and Fact-Checker.
You will be provided with a Claim and an Image.

╔══════════════════════════════════════════════════════════════════╗
║              STRICT VERDICT GUIDELINES (MANDATORY)              ║
╠══════════════════════════════════════════════════════════════════╣
║ 1. NEI IS STRICTLY FORBIDDEN unless the image is COMPLETELY      ║
║    UNRELATED to the claim's subject matter. NEI is last resort.  ║
║                                                                  ║
║ 2. LOGICAL DEDUCTION REQUIRED: Use visual context, scene         ║
║    analysis, and any text visible in the image to reason.        ║
║    Partial visual evidence is SUFFICIENT for a verdict.          ║
║                                                                  ║
║ 3. DIRECTIONAL RULE:                                             ║
║    - Image is consistent with the claim → "Supported"            ║
║    - Image contradicts or is inconsistent → "Refuted"            ║
║                                                                  ║
║ 4. WHEN IN DOUBT: Always choose "Supported" or "Refuted".        ║
╚══════════════════════════════════════════════════════════════════╝

Respond ONLY in valid JSON format:
{"verdict": "Supported" or "Refuted" or "NEI", "confidence": 0.0, "rationale": "short explanation"}"""


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def visual_verifier_aggressive(state: PrismState) -> PrismState:
    """
    [Aggressive 버전] Anti-NEI 강화 Visual Verifier.
    이미지가 없는 경우 NEI 대신 'Supported' 폴백을 사용하여 NEI 투표 누적을 방지합니다.
    (이미지가 없으면 시각적 반박 증거도 없으므로 Supported로 간주)
    """
    print("--- 👁️ [AGGRESSIVE] Running Visual Verifier ---")

    claim = state.get("claim", "This image contains specific identifiable entities or events.")
    image_paths = state.get("image_evidence", [])
    current_verdicts = state.get("verdicts", {})

    # [Aggressive 변경점] 이미지 없음 → NEI 대신 Supported 폴백
    # 근거: 이미지 증거가 없으면 시각적으로 반박할 근거도 없음 → 기본 신뢰
    if not image_paths or not os.path.exists(image_paths[0]):
        print("   [경고] 유효한 이미지 없음 → [Aggressive] NEI 대신 Supported 폴백 처리")
        current_verdicts["visual"] = {
            "verdict": "Supported",
            "confidence": 0.3,
            "rationale": "[Aggressive Mode] No image evidence available; defaulting to Supported (no visual contradiction found)."
        }
        return {"verdicts": current_verdicts}

    target_image_path = image_paths[0]
    base64_image = encode_image(target_image_path)

    user_message = HumanMessage(
        content=[
            {"type": "text", "text": f"Claim: {claim}\nAnalyze the attached image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    )

    print(f"   ⏳ [Aggressive] GPT-4o가 이미지({os.path.basename(target_image_path)})를 분석 중...")
    try:
        response = vision_llm.invoke([SystemMessage(content=_SYSTEM_PROMPT_AGGRESSIVE), user_message])
        clean_content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_content)
    except Exception as e:
        print(f"❌ [Aggressive] Visual Verifier 파싱 에러: {e}")
        result = {"verdict": "NEI", "confidence": 0.0, "rationale": f"Error: {e}"}

    current_verdicts["visual"] = result
    print(f"   ✅ [Aggressive 시각 판정 완료] {result.get('verdict')}")

    return {"verdicts": current_verdicts}
