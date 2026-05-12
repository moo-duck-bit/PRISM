from datasets import load_dataset
import pandas as pd

def load_mocheg_from_hf(repo_id="vcmt794/mocheg-train", split_ratio=50):
    """
    허깅페이스 저장소에서 데이터를 지정된 비율(%)만큼만 다운로드합니다.
    """
    print(f"📥 {repo_id}에서 데이터의 {split_ratio}%를 로드 중...")

    # 슬라이싱 문법 사용: "train[:50%]"
    subset_split = f"train[:{split_ratio}%]"
    
    try:
        # 데이터셋 로드
        dataset = load_dataset(repo_id, split=subset_split)
        
        print(f"✅ 로드 완료! 총 {len(dataset)}개의 샘플을 가져왔습니다.")
        
        # LangGraph에서 다루기 쉽게 리스트(dict) 형태로 변환하여 반환
        return list(dataset)
        
    except Exception as e:
        print(f"❌ 데이터 로드 중 에러 발생: {e}")
        return []

def preview_data(dataset_list):
    """
    로드된 데이터의 구조를 확인하기 위한 간단한 프리뷰 함수
    """
    if not dataset_list:
        return
    
    print("\n🔍 [데이터 구조 미리보기]")
    first_item = dataset_list[0]
    for key, value in first_item.items():
        # 데이터가 너무 길면 잘라서 보여줌
        val_str = str(value)
        print(f"- {key}: {val_str[:100]}{'...' if len(val_str) > 100 else ''}")

if __name__ == "__main__":
    # 테스트: 절반(50%)만 가져오기
    mocheg_data = load_mocheg_from_hf(split_ratio=50)
    
    if mocheg_data:
        preview_data(mocheg_data)