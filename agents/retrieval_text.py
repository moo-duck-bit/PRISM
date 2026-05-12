import os
import pandas as pd
from rank_bm25 import BM25Okapi
from prism.core.state import PrismState

# 1. 실제 코퍼스 파일 경로 설정
corpus_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Corpus3.csv")

print("📚 [BM25] 실제 코퍼스(Corpus3.csv)를 로드하는 중입니다. (약 10~30초 소요)")

try:
    # 2. pandas를 이용해 CSV 파일 안전하게 읽기
    df = pd.read_csv(corpus_path,nrows=1000)
    
    # 3. 'Origin Document' 열에서 텍스트만 리스트로 추출 (결측치는 빈 문자열로 처리)
    real_corpus = df['Origin Document'].fillna("").tolist()
    print(f"✅ 성공: 총 {len(real_corpus):,}개의 실제 증거 문서를 로드했습니다!")
    
except Exception as e:
    print(f"❌ 코퍼스 로드 에러: {e}")
    # 에러 발생 시 시스템이 멈추지 않도록 비상용 데이터 삽입
    real_corpus = ["임시 문서 1", "임시 문서 2"]

# 4. 3만 개의 데이터로 BM25 인덱스 구축 (메모리에 검색 엔진 띄우기)
print("⚙️ [BM25] 검색 인덱스를 구축하고 있습니다...")
tokenized_corpus = [doc.lower().split() for doc in real_corpus]
bm25 = BM25Okapi(tokenized_corpus)
print("🚀 [BM25] 하이브리드 검색 엔진 준비 완료!\n")

def textual_retrieval_agent(state: PrismState) -> PrismState:
    """
    클레임을 바탕으로 코퍼스 중 가장 관련성 높은 증거를 찾아내고,
    LLM 토큰 한도를 넘지 않도록 길이를 안전하게 잘라냅니다.
    """
    print("--- 🔍 Running Textual Retrieval Agent (Real BM25) ---")
    claim = state["claim"]
    
    tokenized_query = claim.lower().split()
    
    # 💡 토큰 방어 1단계: 5개가 아닌 핵심 Top-3 문서만 가져옵니다.
    top_k_evidence = bm25.get_top_n(tokenized_query, real_corpus, n=3)
    
    safe_evidence = []
    for i, doc in enumerate(top_k_evidence):
        doc_str = str(doc)
        
        # 💡 토큰 방어 2단계: 문서 1개당 최대 1000자(약 250토큰)로 자릅니다.
        # 핵심 단어는 보통 기사 앞부분(Lead)에 몰려있으므로 뒤를 날려도 성능에 큰 지장이 없습니다.
        safe_doc = doc_str[:1000] + "\n...(중략)..." if len(doc_str) > 1000 else doc_str
        safe_evidence.append(f"[문서 {i+1}]: {safe_doc}")
        
    print(f"   [검색 완료] 안전하게 정제된 증거 {len(safe_evidence)}건 확보")
    
    return {"text_evidence": safe_evidence}