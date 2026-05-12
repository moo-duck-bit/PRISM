from typing import TypedDict, List, Dict, Any

class PrismState(TypedDict):
    """
    PRISM 파이프라인에서 에이전트 간 공유되는 전역 상태 구조입니다.
    """
    claim: str                          # 검증할 원본 클레임(주장)
    triage_results: Dict[str, bool]     # Triage Agent의 판단 결과 (이미지 필요 여부 등)
    text_evidence: List[str]            # Textual Retrieval이 찾은 텍스트 증거 리스트
    image_evidence: List[str]           # Visual Retrieval이 찾은 이미지 URL 리스트
             # Explanation Agent의 설명문
    # 4개의 Verifier 에이전트가 각자의 판정 결과를 기록할 딕셔너리
    verdicts: Dict[str, Dict[str, Any]]
    explanation: Dict[str, Any]
    final_article: str