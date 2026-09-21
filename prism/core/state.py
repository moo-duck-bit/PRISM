"""파이프라인 전 구간에서 공유되는 상태.

논문 표기와의 대응:
    x = (c, P, O, I)  ->  claim, post_text, ocr_text, image_paths
    m                 ->  modality_mask
    S(m)              ->  active_agents
    E_T, E_I          ->  text_evidence, image_evidence
    v_s = (y_s, g_s, r_s) -> verdicts[s] = {verdict, confidence, rationale}
    y_hat             ->  verdict
    E                 ->  explanation (Toulmin 6요소) + report (렌더링 결과)

LangGraph 노드는 이 딕셔너리를 직접 변경하지 않고, 바꾼 키만 담은 새 딕셔너리를 반환한다.
"""

from typing import Any, Dict, List, TypedDict


class PrismState(TypedDict, total=False):
    # 입력 x = (c, P, O, I)
    claim_id: str
    claim: str
    post_text: str
    ocr_text: str
    image_paths: List[str]

    # Triage 산출물
    modality_mask: Dict[str, bool]   # img, ent, time
    active_agents: List[str]         # S(m). f는 항상 포함된다
    triage_rationale: str

    # 게이팅된 증거
    evidence_pool: List[str]         # D_c. 벤치마크가 증거를 직접 주는 경우 채워진다
    text_evidence: List[str]         # E_T
    image_evidence: List[str]        # E_I

    # 검증기 판정 v_s
    verdicts: Dict[str, Dict[str, Any]]

    # 캘리브레이션 결과
    verdict: str                     # y_hat. 이 값은 이후 단계에서 재계산되지 않는다
    calibration: Dict[str, Any]      # 어떤 규칙이 적용됐는지에 대한 기록

    # 판정과 분리된 설명 E
    explanation: Dict[str, Any]      # Toulmin 6요소
    report: str                      # 6요소를 사람이 읽을 형태로 렌더링한 결과

    metrics: Dict[str, Any]
