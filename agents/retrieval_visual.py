import time
from prism.core.state import PrismState

def visual_retrieval_agent(state: PrismState) -> PrismState:
    """
    클레임을 입력받아 이미지 증거를 검색합니다.
    Triage 결과에서 'image_needed'가 true일 때만 실행됩니다.
    """
    print("--- 📸 Running Visual Retrieval Agent ---")
    _start = time.time()

    triage_results = state.get("triage_results", {})
    metrics = state.get("metrics", {})

    # 1. Triage 관제탑의 지시 확인
    if not triage_results.get("image_needed", False):
        print("   [건너뜀] Triage 결과 이미지 검증이 필요하지 않습니다.")
        metrics["retrieval_visual_latency"] = round(time.time() - _start, 4)
        return {"image_evidence": [], "metrics": metrics}

    print("   [실행됨] 클레임과 관련된 이미지를 검색합니다.")

    # 2. 이미지 검색 로직 (향후 openai/clip-vit-base-patch32 코드가 들어갈 자리입니다)
    mock_images = ["00075-547480-02-GettyImages-959534426.jpg"]

    metrics["retrieval_visual_latency"] = round(time.time() - _start, 4)
    metrics["image_retrieval_calls"] = metrics.get("image_retrieval_calls", 0) + 1

    return {"image_evidence": mock_images, "metrics": metrics}
