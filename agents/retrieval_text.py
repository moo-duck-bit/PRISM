import os
import time
import pandas as pd
from rank_bm25 import BM25Okapi
from prism.core.state import PrismState

# ---------------------------------------------------------------------------
# 코퍼스 로드: Corpus3.csv에서 claim_id별 연결 문서 딕셔너리 구축
#
# MOCHEG 공식 평가 방식과 동일하게, 각 클레임에 실제 연결된 문서만 사용합니다.
# (nrows 제한 없이 전체 로드 후 claim_id 기준 그룹핑)
# ---------------------------------------------------------------------------
corpus_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Corpus3.csv")

print("📚 [BM25] Corpus3.csv 전체 로드 중 (claim_id 기준 인덱싱)...")

try:
    df = pd.read_csv(corpus_path)  # nrows 제한 없이 전체 로드
    df["claim_id"] = df["claim_id"].astype(str)
    df["Origin Document"] = df["Origin Document"].fillna("")

    # claim_id → 연결 문서 리스트 딕셔너리
    _claim_docs: dict[str, list[str]] = (
        df.groupby("claim_id")["Origin Document"]
        .apply(list)
        .to_dict()
    )

    total_claims = len(_claim_docs)
    total_docs   = len(df)
    print(f"✅ 성공: {total_docs:,}개 문서 / {total_claims:,}개 클레임 인덱싱 완료!")

except Exception as e:
    print(f"❌ 코퍼스 로드 에러: {e}")
    _claim_docs = {}

print("🚀 [BM25] claim_id 기반 검색 엔진 준비 완료!\n")


def _bm25_search(docs: list[str], query: str, n: int = 3) -> list[str]:
    """주어진 문서 풀에서 BM25로 Top-n 검색."""
    if not docs:
        return []
    tokenized = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    return bm25.get_top_n(query.lower().split(), docs, n=min(n, len(docs)))


def textual_retrieval_agent(state: PrismState) -> PrismState:
    """
    claim_id에 연결된 MOCHEG 문서 풀에서 BM25로 Top-3 증거를 검색합니다.
    claim_id가 없거나 연결 문서가 없는 경우 전체 코퍼스에서 fallback 검색합니다.
    """
    print("--- 🔍 Running Textual Retrieval Agent (claim_id BM25) ---")
    _start = time.time()

    claim    = state["claim"]
    claim_id = state.get("claim_id", "")

    # 1. claim_id 기반 문서 풀 선택
    candidate_docs = _claim_docs.get(claim_id, [])

    if candidate_docs:
        print(f"   [클레임 연결 문서] claim_id={claim_id}, 후보 {len(candidate_docs)}개")
    else:
        # fallback: 전체 문서에서 검색 (단, 성능 보호를 위해 최대 5000개 샘플)
        print(f"   [Fallback] claim_id={claim_id} 문서 없음 → 전체 코퍼스 검색")
        import random
        all_docs = [doc for docs in _claim_docs.values() for doc in docs]
        candidate_docs = random.sample(all_docs, min(5000, len(all_docs))) if all_docs else []

    # 2. BM25 검색
    top_k_evidence = _bm25_search(candidate_docs, claim, n=3)

    # 3. 토큰 방어: 문서당 최대 1000자
    safe_evidence = []
    for i, doc in enumerate(top_k_evidence):
        doc_str = str(doc)
        safe_doc = doc_str[:1000] + "\n...(중략)..." if len(doc_str) > 1000 else doc_str
        safe_evidence.append(f"[문서 {i+1}]: {safe_doc}")

    print(f"   [검색 완료] 증거 {len(safe_evidence)}건 확보")

    metrics = state.get("metrics", {})
    metrics["retrieval_text_latency"] = round(time.time() - _start, 4)
    metrics["text_retrieval_calls"]   = metrics.get("text_retrieval_calls", 0) + 1

    return {"text_evidence": safe_evidence, "metrics": metrics}
