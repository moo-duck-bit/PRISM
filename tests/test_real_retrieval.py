import os
import sys

# 프로젝트 루트 경로를 시스템 패스에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from prism.agents.retrieval_text import textual_retrieval_agent
from prism.core.state import PrismState

def run_retrieval_test():
    print("=== 🚀 진짜 코퍼스(40만 개) 검색 테스트 시작 ===\n")
    
    # 가상의 클레임 생성 (아까 데이터 첫 줄에 있던 내용 기반)
    mock_state: PrismState = {
        "claim": "Ellen Weintraub talks about voter fraud.",
        "triage_results": {},
        "text_evidence": [],
        "image_evidence": [],
        "verdicts": {}
    }
    
    # 에이전트 실행 (이때 corpus가 로드되고 인덱스가 구축됩니다)
    new_state = textual_retrieval_agent(mock_state)
    
    print("\n✅ [검색된 Top-5 증거 미리보기]")
    for i, doc in enumerate(new_state.get("text_evidence", []), 1):
        # 텍스트가 너무 기니까 200자까지만 잘라서 보여줍니다.
        doc_str = str(doc).replace('\n', ' ')
        print(f"\n[{i}위] {doc_str[:200]}...")

if __name__ == "__main__":
    run_retrieval_test()