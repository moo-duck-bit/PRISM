import os
import sys
import glob

# 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from prism.agents.verifier_visual import visual_verifier
from prism.core.state import PrismState

def run_safe_visual_test():
    print("=== 👁️ 토큰 방어형: 시각 검증기 3장 테스트 ===\n")
    
    # 이미지 폴더 경로 (LAMDALab 서버 기준)
    base_dir = os.path.dirname(os.path.dirname(__file__))
    image_dir = os.path.join(base_dir, "data", "mocheg", "images")
    
    # 폴더 안에서 .jpg 또는 .jpeg 파일 딱 3개만 선착순으로 찾기
    surviving_images = glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.jpeg"))
    test_images = surviving_images[:3]
    
    if not test_images:
        print(f"❌ '{image_dir}' 폴더에 이미지가 없습니다. 경로를 다시 확인해 주세요!")
        return
        
    print(f"✅ 총 {len(test_images)}장의 이미지를 테스트합니다. (안전 모드 가동)\n")
    print("-" * 60)
    
    # 3장의 이미지를 하나씩 GPT-4o에게 먹여서 확인
    for i, img_path in enumerate(test_images, 1):
        print(f"[{i}번째 이미지 검증 중]: {os.path.basename(img_path)}")
        
        # 가상의 상태(State) 생성 (이미지에 대한 범용적인 클레임 적용)
        state: PrismState = {
            "claim": "This image contains specific identifiable entities or events.",
            "triage_results": {"visual_needed": True},
            "text_evidence": [],
            "image_evidence": [img_path],
            "verdicts": {}
        }
        
        # 시각 검증 에이전트 실행
        state.update(visual_verifier(state))
        
        # 최종 판정 결과 출력
        result = state.get("verdicts", {}).get("visual", {})
        print(f"   👉 [판정] {result.get('verdict')} (확신도: {result.get('confidence')})")
        print(f"   👉 [이유] {result.get('rationale')}")
        print("-" * 60)

if __name__ == "__main__":
    run_safe_visual_test()