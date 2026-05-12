# PRISM: Multimodal Fact-Checking Pipeline
## System Specification Document (SPEC.md)

> **Version**: 0.3.0 | **Updated**: 2026-05-12  
> **Target Venue**: CIKM 2026 Short Paper (8 pages, 4,000 words)  
> **Architecture**: LangGraph-based Multi-Agent System  
> **Lead**: Microsoft AI Research Lab

---

## Table of Contents

1. [Project Goals & CIKM Targeting](#1-project-goals--cikm-targeting)
2. [System Architecture](#2-system-architecture)
3. [Agent Delegation & Parallelization](#3-agent-delegation--parallelization)
4. [Evaluation & Testing Pipeline](#4-evaluation--testing-pipeline)
5. [Edge Case Definitions & Handling Logic](#5-edge-case-definitions--handling-logic)
6. [Cost Model & Token Budget](#6-cost-model--token-budget)

---

## 1. Project Goals & CIKM Targeting

### 1-1. 핵심 연구 목적 (Research Objective)

PRISM은 자연어 클레임(Claim)과 연관 이미지를 함께 분석하여 **"Supported / Refuted / NEI"** 판정을 자동으로 수행하는 멀티모달 팩트체킹 파이프라인이다.

**기존 한계 (Motivation)**:
- 기존 팩트체킹 시스템은 텍스트 단일 모달에 집중 → 이미지 조작(deepfake, 맥락 오용) 탐지 불가
- 단일 LLM 프롬프트 기반 접근은 역할 혼합으로 인한 환각(hallucination) 발생률 높음
- 검증 근거(rationale)의 투명성 부재 → 저널리즘 신뢰성 문제

**PRISM의 기여 (Contribution)**:
1. **분업화된 Multi-Agent 검증**: 텍스트·개체·시공간·시각 검증을 독립 에이전트로 분리
2. **Toulmin 논증 모델 기반 보고서 생성**: 구조화된 팩트체킹 리포트 자동 작성
3. **BM25 하이브리드 검색**: Real Corpus (MOCHEG 2023) 기반의 실제 증거 검색
4. **비용 효율적 LLM 라우팅**: 태스크 복잡도에 따른 gpt-4o-mini / gpt-4o 분리 사용

### 1-2. CIKM Short Paper 타겟팅 전략

| 항목 | 목표 |
|------|------|
| **페이퍼 타입** | Short Research Paper (4~6 pages) |
| **핵심 주장** | 역할 분리된 Multi-Agent가 단일 LLM 대비 팩트체킹 정확도 향상 |
| **평가 지표** | Accuracy, F1 (3-class), NEI Rate, Cost per Sample |
| **Ablation Study** | 에이전트 제거 실험 (w/o Visual, w/o Entity, w/o ST) |
| **데이터셋** | MOCHEG 2023 (멀티모달 클레임 검증 벤치마크) |
| **비교 베이스라인** | Single-LLM, BERT-NLI, Dual-Encoder Retriever |

---

## 2. System Architecture

### 2-1. 전체 파이프라인 흐름도

```
INPUT: { claim: str, image_path: str | None }
              │
              ▼
    ┌─────────────────┐
    │  retrieve Node  │  ← BM25 + Dense Hybrid Search
    │  (text + image) │    Corpus: MOCHEG Corpus3.csv
    └────────┬────────┘
             │ text_evidence[], image_evidence[]
             ▼
    ┌─────────────────┐
    │ verify_factual  │  ← gpt-4o-mini | max_tokens=100
    └────────┬────────┘    verdict["factual"]
             ▼
    ┌─────────────────┐
    │ verify_entity   │  ← spaCy NER + gpt-4o-mini
    └────────┬────────┘    verdict["entity"]
             ▼
    ┌──────────────────────────┐
    │ verify_spatiotemporal    │  ← spaCy + gpt-4o-mini
    └────────┬─────────────────┘    verdict["spatiotemporal"]
             ▼
    ┌─────────────────┐
    │ verify_visual   │  ← gpt-4o Vision | max_tokens=150
    └────────┬────────┘    verdict["visual"]
             ▼
    ┌───────────────────────┐
    │ generate_article      │  ← gpt-4o | max_tokens=2000
    │ (Toulmin Generator)   │    [ON/OFF 전략 §4-3 참고]
    └────────┬──────────────┘
             ▼
OUTPUT: { verdicts: dict, final_article: str | None }
```

### 2-2. PrismState 스키마

```python
class PrismState(TypedDict):
    claim: str                          # 검증 대상 클레임
    triage_results: Dict[str, bool]     # 라우팅 지시 (이미지 필요 여부 등)
    text_evidence: List[str]            # 텍스트 검색 결과 (top-k 문단)
    image_evidence: List[str]           # 이미지 경로 리스트
    verdicts: Dict[str, Dict[str, Any]] # 에이전트별 판정 결과
    explanation: Dict[str, Any]         # Explanation 에이전트 결과
    final_article: str                  # Toulmin 보고서 (Optional)
```

**Verdict 스키마 (에이전트 공통)**:
```json
{
  "verdict": "Supported" | "Refuted" | "NEI",
  "confidence": 0.0 ~ 1.0,
  "rationale": "짧은 근거 설명 (영문, 50자 이내)"
}
```

---

## 3. Agent Delegation & Parallelization

### 3-1. 에이전트 역할 분담표

| Agent | 파일 | 모달리티 | 사용 도구 | 출력 |
|-------|------|----------|-----------|------|
| **Textual Retrieval** | `retrieval_text.py` | 텍스트 | BM25, FAISS | `text_evidence[]` |
| **Visual Retrieval** | `retrieval_visual.py` | 이미지 | CLIP, FAISS | `image_evidence[]` |
| **Factual Verifier** | `verifier_factual.py` | 텍스트 | gpt-4o-mini | `verdicts["factual"]` |
| **Entity Verifier** | `verifier_entity.py` | 텍스트 | spaCy NER + LLM | `verdicts["entity"]` |
| **Spatiotemporal Verifier** | `verifier_spatiotemporal.py` | 텍스트 | spaCy + LLM | `verdicts["spatiotemporal"]` |
| **Visual Verifier** | `verifier_visual.py` | 이미지 | gpt-4o Vision | `verdicts["visual"]` |
| **Toulmin Generator** | `generator_toulmin.py` | 텍스트 | gpt-4o | `final_article` |

### 3-2. 병렬화 전략 (Future: v0.4.0)

현재(v0.3) 파이프라인은 순차 실행이나, 다음 버전에서 LangGraph `Send` API를 활용한 병렬화를 계획한다.

```
retrieve
    │
    ├──────────────────────────────┐
    │                              │
    ▼                              ▼
[verify_factual]         [verify_visual]      ← 병렬 실행 (이미지/텍스트 독립)
[verify_entity]
[verify_spatiotemporal]
    │                              │
    └──────────────────────────────┘
                    │
                    ▼
           [generate_article]
```

**병렬화 조건**: `text_evidence`와 `image_evidence`는 서로 독립적이므로, 
텍스트 검증 3종 + 시각 검증 1종을 `fan-out → fan-in` 패턴으로 동시 실행 가능.

### 3-3. Triage 라우팅 로직

`triage_results` 딕셔너리를 통해 에이전트를 조건부 실행한다:

```python
# triage_results 예시
{
    "factual_needed": True,       # 텍스트 검증 필요 여부
    "entity_needed": True,        # 개체명 검증 필요 여부
    "spatiotemporal_needed": True, # 시공간 검증 필요 여부
    "visual_needed": False,       # 이미지가 없는 경우 False
    "article_needed": True        # 보고서 생성 필요 여부 (비용 제어)
}
```

---

## 4. Evaluation & Testing Pipeline

### 4-1. 평가 지표 (Metrics)

**Primary Metrics**:
| 지표 | 설명 | 계산 방법 |
|------|------|----------|
| **Accuracy** | 전체 3-class 분류 정확도 | `(TP_S + TP_R + TP_N) / Total` |
| **Macro F1** | 클래스 불균형 보정 F1 | `mean(F1_Supported, F1_Refuted, F1_NEI)` |
| **NEI Rate** | NEI 판정 비율 (보수성 지표) | `count(NEI) / Total` |
| **Cost/Sample** | 샘플당 평균 API 비용 | `total_token_cost / N` |

**Secondary Metrics (Retrieval)**:
- `Recall@5`, `MRR@10` (텍스트 검색 품질)
- `Visual Relevance Score` (이미지-클레임 연관성)

### 4-2. 배치 실험 계획 (CI/CD & TDD 관점)

```
데이터셋: MOCHEG 2023 (train.json)
  ├─ 개발용 mini-batch: 50 samples (빠른 smoke test)
  ├─ 검증용 dev-batch:  200 samples (에이전트별 단위 평가)
  └─ 논문용 full-batch: 1,000 samples (최종 결과 보고)
```

**TDD 단계별 테스트 전략**:

```
Phase 1: Unit Tests (tests/ 폴더)
  └─ test_factual_verifier.py     ← 증거 없음 → NEI 반환 확인
  └─ test_entity_verifier.py      ← spaCy 파싱 실패 시 fallback 확인
  └─ test_visual_verifier.py      ← 이미지 없음 → NEI 반환 확인
  └─ test_toulmin_generator.py    ← Toulmin 구조 키워드 포함 여부 확인

Phase 2: Integration Tests
  └─ test_pipeline_smoke.py       ← 단일 샘플 end-to-end 실행
  └─ test_pipeline_no_image.py    ← 이미지 없는 텍스트 전용 클레임

Phase 3: Batch Evaluation
  └─ evaluation/verification_eval.py  ← 배치 정확도/F1 자동 계산
  └─ evaluation/cost_tracker.py       ← 토큰 비용 자동 집계 (TODO)
```

### 4-3. Article Generation 노드 On/Off 전략

`generate_article` 노드는 **비용이 가장 높은 노드** (gpt-4o, max_tokens=2000)이다.
평가 실험 시에는 Off하여 비용을 절감한다.

**전략 A: 환경 변수 제어**
```python
# configs/config.yaml
pipeline:
  generate_article: false   # 평가 시 OFF, 데모/서비스 시 ON
```

**전략 B: 조건부 라우팅 (권장)**
```python
# core/graph.py
def should_generate_article(state: PrismState) -> str:
    if state.get("triage_results", {}).get("article_needed", False):
        return "generate_article"
    return END

workflow.add_conditional_edges("verify_visual", should_generate_article)
```

**비용 절감 효과 추정**:
| 실험 모드 | 노드 구성 | 예상 Cost/1000 samples |
|----------|----------|----------------------|
| Eval-Only | retrieve + 4 verifiers | ~$2.5 |
| Full Pipeline | + generate_article | ~$18.0 |
| Mini Eval (50) | Eval-Only | ~$0.12 |

### 4-4. Ablation Study 설계

**목표**: 각 에이전트의 개별 기여도 측정 (CIKM 논문 필수 항목)

| 실험 조건 | 비활성화 에이전트 | 가설 |
|----------|-----------------|------|
| **Full PRISM** | 없음 | 최고 성능 |
| **w/o Visual** | `verify_visual` 제거 | 이미지 클레임 정확도 하락 예상 |
| **w/o Entity** | `verify_entity` 제거 | 인물/장소 클레임 정확도 하락 예상 |
| **w/o ST** | `verify_spatiotemporal` 제거 | 날짜/위치 클레임 정확도 하락 예상 |
| **Text-Only** | Visual + Entity + ST 제거 | 기존 단일 모달 시스템과 동급 예상 |
| **Single-LLM** | 모든 verifier → 단일 프롬프트 | 환각 증가 예상 |

---

## 5. Edge Case Definitions & Handling Logic

### 5-1. 이미지가 클레임과 무관한 경우 (Irrelevant Image)

**정의**: `image_evidence`에 이미지가 존재하나, 클레임과 시각적으로 관련 없는 경우.

**탐지 조건**:
```python
# verifier_visual.py 내 처리 로직
if result.get("confidence", 0.0) < 0.3 and result.get("verdict") == "NEI":
    # 이미지가 클레임과 무관한 것으로 판단
    result["rationale"] = "Image appears unrelated to the claim. Skipping visual verification."
    result["verdict"] = "NEI"
    result["image_relevant"] = False  # 추가 메타데이터
```

**파이프라인 처리 흐름**:
```
verify_visual
  ├─ confidence >= 0.3 → 정상 판정 (Supported/Refuted/NEI)
  ├─ confidence < 0.3 AND verdict="NEI" → 이미지 무관 처리
  │     └─ verdicts["visual"]["image_relevant"] = False
  │     └─ Generator: 이미지 증거 섹션 생략
  └─ 이미지 파일 없음 → NEI 즉시 반환 (§3 참고)
```

**Generator 대응**:
```python
# generator_toulmin.py
visual_result = verdicts.get("visual", {})
if not visual_result.get("image_relevant", True):
    # [Data/Grounds] 섹션에서 이미지 증거 언급 제외
    image_section = "[이미지 증거가 클레임과 무관하여 시각 검증이 제외되었습니다.]"
```

### 5-2. 텍스트 검색 결과가 엉뚱한 경우 (Off-Topic Retrieval)

**정의**: BM25 검색이 반환한 `text_evidence`가 클레임과 주제적으로 관련 없는 경우.

**탐지 조건 (Soft Check)**:
```python
# verifier_factual.py 내 처리 로직
# LLM이 증거와 클레임의 연관성이 낮다고 판단하는 경우
if result.get("confidence", 0.0) < 0.2:
    # 검색 결과가 클레임과 무관할 가능성 높음
    result["verdict"] = "NEI"
    result["retrieval_quality"] = "low"
```

**탐지 조건 (Hard Check - 권장)**:
```python
# retrieval_text.py 내 검색 후처리
def filter_irrelevant_evidence(claim: str, evidence_list: List[str]) -> List[str]:
    """BM25 점수 임계값 미달 증거 필터링"""
    # BM25 score < threshold 이면 제거
    RELEVANCE_THRESHOLD = 5.0  # 실험적으로 조정
    filtered = [e for e, score in zip(evidence_list, scores) if score >= RELEVANCE_THRESHOLD]
    if not filtered:
        return []  # 빈 리스트 → 각 verifier에서 NEI 처리
    return filtered[:3]  # top-3 만 사용
```

**처리 우선순위**:
1. **Hard Check (검색 단계)**: BM25 점수 임계값 필터링 → 검색 단계에서 차단
2. **Soft Check (검증 단계)**: LLM confidence 낮으면 NEI → 환각 방지
3. **Fallback**: 어떤 경우에도 NEI는 허용 — 오판(false claim)보다 보수적 판정이 우선

### 5-3. 클레임에 이미지가 없는 경우 (Text-Only Claim)

```python
# triage.py (또는 main.py 진입점)
def prepare_state(claim: str, image_path: str | None) -> PrismState:
    return PrismState(
        claim=claim,
        triage_results={
            "visual_needed": image_path is not None,
            "article_needed": True,
        },
        text_evidence=[],
        image_evidence=[image_path] if image_path else [],
        verdicts={},
        explanation={},
        final_article=""
    )
```

**처리 결과**: `triage_results["visual_needed"] = False` → `verify_visual`은 즉시 NEI 반환 후 종료.

### 5-4. 모든 에이전트가 NEI를 반환하는 경우 (All-NEI)

```python
# generator_toulmin.py 내 처리
all_nei = all(v.get("verdict") == "NEI" for v in verdicts.values())
if all_nei:
    return {
        "final_article": (
            "[검증 불가] 모든 검증 에이전트가 충분한 증거를 찾지 못했습니다. "
            "이 클레임은 현재 데이터베이스로 검증할 수 없습니다."
        )
    }
```

---

## 6. Cost Model & Token Budget

### 6-1. 샘플당 예상 토큰 비용

| 노드 | 모델 | 입력 토큰 (est.) | 출력 토큰 (max) | 비용/샘플 (est.) |
|------|------|----------------|----------------|-----------------|
| `retrieve` | - | - | - | $0.000 |
| `verify_factual` | gpt-4o-mini | ~300 | 100 | $0.000060 |
| `verify_entity` | gpt-4o-mini | ~200 | 100 | $0.000045 |
| `verify_spatiotemporal` | gpt-4o-mini | ~200 | 100 | $0.000045 |
| `verify_visual` | gpt-4o | ~200 + image | 150 | $0.001200 |
| `generate_article` | gpt-4o | ~500 | 2000 | $0.015000 |
| **합계 (Full)** | | | | **~$0.0164** |
| **합계 (Eval-Only)** | | | | **~$0.0015** |

> 가격 기준: gpt-4o-mini $0.15/1M input + $0.60/1M output, gpt-4o $2.50/1M input + $10.00/1M output (2026-05 기준)

### 6-2. 월별 실험 예산 계획

| 실험 규모 | 샘플 수 | 예상 비용 (Eval-Only) | 예상 비용 (Full) |
|----------|--------|---------------------|----------------|
| Smoke Test | 50 | ~$0.075 | ~$0.82 |
| Dev Batch | 200 | ~$0.30 | ~$3.28 |
| Paper Batch | 1,000 | ~$1.50 | ~$16.40 |
| Ablation (6×) | 6,000 | ~$9.00 | - (Eval-Only) |

---

*This document is the single source of truth for PRISM architecture decisions.*  
*All implementation must conform to this spec. Deviations require a spec update first.*
