"""Gated Evidence Retrieval (논문 3.3절, 텍스트 경로).

오픈 웹 검색은 노이즈 주입과 정보 오염에 취약하다. PRISM은 검색을 클레임에
연결된 신뢰 증거 풀 D_c 로 제한하고, 그 안에서만 BM25 상위 k건을 고른다.

    E_T = TopK_{k=3}(BM25(D_c, c))

D_c 는 벤치마크마다 다르다.
- MOCHEG: claim_id 로 연결된 Corpus3 문서 (문서 단위)
- RW-Post: 벤치마크가 제공하는 gold fact-check 증거 (스니펫 단위)

gated=False 는 게이트를 제거한 closed-book 설정이다. 증거 게이팅의 기여도를
분리 측정하기 위한 ablation 경로이며, 이 경우 E_T = 공집합이 된다.
"""

import logging
import os
import random
import time
from functools import lru_cache
from typing import Dict, List

import pandas as pd
from rank_bm25 import BM25Okapi

from prism.core.state import PrismState

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS_PATH = os.getenv(
    "PRISM_CORPUS_PATH",
    os.path.join(_PROJECT_ROOT, "data", "Corpus3.csv"),
)

TOP_K = 3
BM25_K1 = 1.5
BM25_B = 0.75

MAX_DOC_CHARS = 1000        # 문서 하나가 프롬프트를 독점하지 않도록 자른다
FALLBACK_POOL_SIZE = 5000   # 연결 문서가 없을 때 인덱싱 비용 상한
FALLBACK_SEED = 42          # 폴백 표본을 재현 가능하게 고정


@lru_cache(maxsize=1)
def claim_documents() -> Dict[str, List[str]]:
    """D_c 인덱스. claim_id -> 연결 문서 리스트. 최초 호출 시 한 번만 읽는다."""
    try:
        df = pd.read_csv(CORPUS_PATH)
    except (OSError, pd.errors.ParserError) as exc:
        logger.error("코퍼스 로드 실패 (%s): %s", CORPUS_PATH, exc)
        return {}

    df["claim_id"] = df["claim_id"].astype(str)
    df["Origin Document"] = df["Origin Document"].fillna("")

    index = df.groupby("claim_id")["Origin Document"].apply(list).to_dict()
    logger.info("D_c 인덱싱 완료: 문서 %d개 / 클레임 %d개", len(df), len(index))
    return index


def bm25_search(documents: List[str], query: str, top_n: int = TOP_K) -> List[str]:
    """문서 풀에서 BM25 상위 top_n건을 돌려준다.

    후보 집합이 클레임마다 달라지므로 인덱스를 미리 만들어 둘 수 없고 호출마다 구축한다.
    클레임당 후보가 수십 건 수준이라 비용은 무시할 만하다.
    """
    if not documents:
        return []
    bm25 = BM25Okapi(
        [doc.lower().split() for doc in documents],
        k1=BM25_K1,
        b=BM25_B,
    )
    return bm25.get_top_n(query.lower().split(), documents, n=min(top_n, len(documents)))


def format_evidence(documents: List[str], max_chars: int = MAX_DOC_CHARS) -> List[str]:
    """프롬프트에 넣을 수 있도록 번호를 붙이고 길이를 제한한다."""
    formatted = []
    for i, doc in enumerate(documents, start=1):
        text = str(doc)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...(truncated)..."
        formatted.append(f"[E{i}] {text}")
    return formatted


def _fallback_pool(index: Dict[str, List[str]]) -> List[str]:
    all_docs = [doc for docs in index.values() for doc in docs]
    if len(all_docs) <= FALLBACK_POOL_SIZE:
        return all_docs
    return random.Random(FALLBACK_SEED).sample(all_docs, FALLBACK_POOL_SIZE)


def textual_retrieval_agent(state: PrismState, gated: bool = True) -> PrismState:
    """E_T 를 채운다. gated=False면 게이트를 제거한 closed-book 경로다."""
    started = time.time()
    metrics = state.get("metrics", {})

    if not gated:
        logger.info("closed-book: 외부 증거 없이 진행")
        metrics["retrieval_text_latency"] = round(time.time() - started, 4)
        metrics["text_retrieval_calls"] = metrics.get("text_retrieval_calls", 0)
        return {"text_evidence": [], "metrics": metrics}

    # 벤치마크가 증거 스니펫을 직접 주는 경우(RW-Post)는 그 풀 안에서 검색한다.
    candidates = list(state.get("evidence_pool", []) or [])
    source = "state_pool"

    if not candidates:
        index = claim_documents()
        claim_id = state.get("claim_id", "")
        candidates = index.get(claim_id, [])
        source = "claim_linked"
        if not candidates:
            candidates = _fallback_pool(index)
            source = "corpus_fallback"

    evidence = format_evidence(bm25_search(candidates, state["claim"]))
    logger.info("E_T: %d건 (source=%s, 후보 %d)", len(evidence), source, len(candidates))

    metrics["retrieval_text_latency"] = round(time.time() - started, 4)
    metrics["text_retrieval_calls"] = metrics.get("text_retrieval_calls", 0) + 1
    metrics["evidence_source"] = source

    return {"text_evidence": evidence, "metrics": metrics}
