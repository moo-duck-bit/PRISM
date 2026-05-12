import os
import sys

# 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from prism.core.graph import prism_app

def run_e2e_test():
    print("=== 🌟 PRISM 멀티모달 통합(E2E) 테스트 시작 ===\n")
    
    # 아까 성공했던 코로나 백신 사진 경로 (경로를 진경님 서버에 맞게 설정)
    base_dir = os.path.dirname(os.path.dirname(__file__))
    vaccine_image = os.path.join(base_dir, "data", "mocheg", "images", "103776-130235-01-COVID-Vaccine_Course-Card.jpg")
    
    # 입력할 데이터 (클레임과 이미지)
    initial_state = {
        "claim": "COVID-19 vaccines are being distributed and prepared in vials.",
        "triage_results": {
            "factual_needed": True,
            "entity_needed": True,
            "spatiotemporal_needed": True,
            "visual_needed": True  # 시각 검증 ON!
        },
        "text_evidence": [],
        "image_evidence": [vaccine_image], # 백신 사진 투입
        "verdicts": {},
        "final_article": ""
    }
    
    # 🚀 랭그래프 엔진 가동! (우리가 만든 모든 에이전트가 알아서 순서대로 돕니다)
    print("🚀 팩트체킹 엔진 작동 시작...\n")
    final_state = prism_app.invoke(initial_state)
    
    print("\n" + "="*60)
    print("📜 [최종 생성된 멀티모달 팩트체킹 보고서]")
    print("="*60)
    print(final_state.get("final_article", "기사 생성 실패"))
    print("="*60)

if __name__ == "__main__":
    run_e2e_test()