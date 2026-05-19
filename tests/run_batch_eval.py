"""
CIKM 논문 제출용 PRISM 배치 평가 스크립트 (Resume + Per-row Append 지원)

[샘플링 정책 — 논문 Table 1 기준]
- 소스: sample_mocheg_v2.py 가 생성한 sampled_300_v2.csv (seed=42, 검증 완료)
- 구성: PolitiFact 클레임만, NEI 제외, Supported 150 + Refuted 150 = 300개
- 재현성: verify_reproducibility_v2.py 로 PASS 확인된 canonical 목록 그대로 사용
          → 이 스크립트는 추가 샘플링을 일절 수행하지 않는다.

[지표]
- Token(비용), Latency(E2E + 에이전트별), Retrieval Count, Explanation Quality (LLM-as-a-Judge)

[Resume 전략]
- cikm_final_metrics.csv 가 존재하면 이미 처리된 claim_id 를 건너뛰고 이어서 실행
- 클레임 1개 처리 완료 시마다 즉시 append 저장 → 서버 강제 종료 시에도 데이터 유실 없음
- API 에러 발생 시 10초 대기 후 1회 재시도 → 재시도도 실패하면 안전 종료(sys.exit)
  → 다시 실행하면 실패한 그 인덱스부터 자동으로 이어서 시작됨
"""

import csv
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

# 프로젝트 루트를 sys.path에 추가 (prism.* 임포트 가능하도록)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from prism.core.graph import prism_app
from prism.agents.triage import triage_agent

# ---------------------------------------------------------------------------
# 상수 설정
# ---------------------------------------------------------------------------
# sample_mocheg_v2.py 가 생성한 검증 완료 파일 (seed=42, 150+150)
SAMPLED_CSV  = "/home/jkpark/prism/sampled_300_v2.csv"
# Claim 텍스트 원본 (Corpus2.csv 의 'Claim' 컬럼에서 로드)
CORPUS2_CSV  = "/home/jkpark/prism/mocheg_json_csv/mocheg/test/Corpus2.csv"

# 단일 출력 파일 — 중간 저장도 여기에 append 하므로 별도 mid 파일 불필요
OUTPUT_FINAL = "cikm_final_metrics.csv"
JUDGE_MODEL  = "gpt-4o"

# 결과 컬럼 순서 (헤더 일관성 보장)
OUTPUT_COLUMNS = [
    "idx", "claim_id", "claim", "ground_truth", "pred_label",
    "total_tokens", "prompt_tokens", "completion_tokens", "total_cost_usd",
    "e2e_latency_sec",
    "triage_latency", "retrieval_text_latency", "retrieval_visual_latency",
    "factual_verifier_latency", "generator_latency",
    "text_retrieval_calls", "image_retrieval_calls",
    "hallucination_free", "claim_support_consistency",
]


# ---------------------------------------------------------------------------
# 기능 A: 최종 판정 집계 (다수결)
# ---------------------------------------------------------------------------

def _aggregate_verdict(verdicts: dict) -> str:
    """
    각 검증 에이전트의 verdict 를 다수결로 집계하여 최종 레이블을 반환합니다.
    투표 동수일 경우 'NEI' 로 처리합니다.

    반환: "Supported" | "Refuted" | "NEI"
    """
    votes = [v.get("verdict", "NEI") for v in verdicts.values() if isinstance(v, dict)]
    if not votes:
        return "NEI"
    cnt = Counter(votes)
    top_label, top_count = cnt.most_common(1)[0]
    # 동수(tie)면 NEI 우선
    if list(cnt.values()).count(top_count) > 1:
        return "NEI"
    return top_label


# ---------------------------------------------------------------------------
# 기능 B: LLM-as-a-Judge 평가 함수
# ---------------------------------------------------------------------------
_judge_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0)


def evaluate_explanation_quality(claim: str, explanation: str, ground_truth: str) -> dict:
    """
    gpt-4o 를 Judge 로 사용하여 explanation 의 품질을 평가합니다.

    반환:
        {
            "hallucination_free": 0 or 1,        # 환각/거짓 정보 없음 여부
            "claim_support_consistency": 0 or 1  # 최종 판정과 논리적 일치 여부
        }
    """
    prompt = f"""You are an expert fact-checking evaluator. Assess the following explanation strictly.

Claim: {claim}
Ground Truth Label: {ground_truth}
Generated Explanation: {explanation}

Evaluate the explanation on TWO criteria and respond ONLY in valid JSON:
1. "hallucination_free": 1 if the explanation contains NO fabricated or false information, 0 otherwise.
2. "claim_support_consistency": 1 if the explanation's final verdict is logically consistent with the Ground Truth Label, 0 otherwise.

Respond ONLY with: {{"hallucination_free": <0 or 1>, "claim_support_consistency": <0 or 1>}}"""

    try:
        response = _judge_llm.invoke(prompt)
        raw = response.content.strip()
        if "```" in raw:
            raw = raw.split("```")[-2].replace("json", "").strip()
        result = json.loads(raw)
        return {
            "hallucination_free": int(result.get("hallucination_free", 0)),
            "claim_support_consistency": int(result.get("claim_support_consistency", 0)),
        }
    except Exception as e:
        print(f"   [Judge 에러] {e}")
        return {"hallucination_free": 0, "claim_support_consistency": 0}


# ---------------------------------------------------------------------------
# 기능 C: 데이터 로드
# ---------------------------------------------------------------------------

def _load_samples() -> list[dict]:
    """
    sampled_300_v2.csv (검증 완료 canonical 목록) + Corpus2.csv 의 Claim 텍스트를 결합하여
    {'claim_id', 'claim', 'label'} 딕셔너리 리스트로 반환합니다.

    추가 샘플링/셔플을 일절 수행하지 않습니다 (Resume 순서 일관성 보장).
    """
    # ── 1. Corpus2.csv 에서 claim_id → Claim 텍스트 매핑 구축 ──────────────────
    if not os.path.exists(CORPUS2_CSV):
        raise FileNotFoundError(f"Corpus2.csv를 찾을 수 없습니다: {CORPUS2_CSV}")

    claim_text: dict[str, str] = {}
    with open(CORPUS2_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row["claim_id"].strip()
            if cid not in claim_text:
                claim_text[cid] = row["Claim"].strip()

    # ── 2. 검증 완료 CSV 에서 순서 그대로 300개 로드 ───────────────────────────
    if not os.path.exists(SAMPLED_CSV):
        raise FileNotFoundError(
            f"sampled_300_v2.csv를 찾을 수 없습니다: {SAMPLED_CSV}\n"
            "먼저 sample_mocheg_v2.py 를 실행하여 canonical 샘플 목록을 생성해 주세요."
        )

    samples: list[dict] = []
    missing: list[str] = []
    with open(SAMPLED_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row["claim_id"].strip()
            lbl = row["label"].strip()
            text = claim_text.get(cid, "")
            if not text:
                missing.append(cid)
                continue
            samples.append({"claim_id": cid, "claim": text, "label": lbl})

    if missing:
        print(f"   ⚠️  Corpus2.csv에서 Claim 텍스트를 찾지 못한 claim_id: {len(missing)}개")
        print(f"      (예시: {missing[:3]})")

    # ⚠️ 절대 셔플하지 않음 — Resume 시 순서 일관성이 보장되어야 함
    return samples


# ---------------------------------------------------------------------------
# 기능 D: 단일 클레임 처리 (성공 시 row dict 반환, 실패 시 예외 raise)
# ---------------------------------------------------------------------------

def _process_one(idx: int, sample: dict, e2e_start: float) -> dict:
    """
    한 개의 클레임을 PRISM 파이프라인으로 처리하고 결과 딕셔너리를 반환합니다.
    내부 에러는 호출자(run_cikm_experiment)가 핸들링하도록 그대로 raise 합니다.
    """
    claim        = sample["claim"]
    ground_truth = sample["label"]

    triage_input = {
        "claim_id": sample["claim_id"],
        "claim": claim,
        "triage_results": {},
        "text_evidence": [],
        "image_evidence": [],
        "verdicts": {},
        "explanation": {},
        "final_article": "",
        "metrics": {},
    }

    try:
        triage_out = triage_agent(triage_input)
        raw_triage = triage_out.get("triage_results", {})
    except Exception as e:
        print(f"   [Triage 에러] {e} → 기본값 사용")
        triage_out = {"metrics": {}}
        raw_triage = {}

    triage_results = {
        "image_needed":          raw_triage.get("image_needed", False),
        "factual_needed":        True,
        "entity_needed":         raw_triage.get("entity", True),
        "spatiotemporal_needed": raw_triage.get("temporal", True),
    }

    print(f"   [Triage] image={triage_results['image_needed']} | "
          f"entity={triage_results['entity_needed']} | "
          f"temporal={triage_results['spatiotemporal_needed']}")

    initial_state = {
        "claim_id": sample["claim_id"],
        "claim": claim,
        "triage_results": triage_results,
        "text_evidence": [],
        "image_evidence": [],
        "verdicts": {},
        "explanation": {},
        "final_article": "",
        "metrics": triage_out.get("metrics", {}),
    }

    with get_openai_callback() as cb:
        final_state = prism_app.invoke(initial_state)

    e2e_latency   = round(time.time() - e2e_start, 4)
    agent_metrics = final_state.get("metrics", {})
    final_article = final_state.get("final_article", "")
    verdicts      = final_state.get("verdicts", {})
    final_verdict = _aggregate_verdict(verdicts)   # pred_label 에 할당

    judge_scores = evaluate_explanation_quality(claim, final_article, ground_truth)

    return {
        "idx": idx,
        "claim_id": sample["claim_id"],
        "claim": claim,
        "ground_truth": ground_truth,
        "pred_label": final_verdict,
        "total_tokens": cb.total_tokens,
        "prompt_tokens": cb.prompt_tokens,
        "completion_tokens": cb.completion_tokens,
        "total_cost_usd": round(cb.total_cost, 6),
        "e2e_latency_sec": e2e_latency,
        "triage_latency": agent_metrics.get("triage_latency"),
        "retrieval_text_latency": agent_metrics.get("retrieval_text_latency"),
        "retrieval_visual_latency": agent_metrics.get("retrieval_visual_latency"),
        "factual_verifier_latency": agent_metrics.get("factual_verifier_latency"),
        "generator_latency": agent_metrics.get("generator_latency"),
        "text_retrieval_calls": agent_metrics.get("text_retrieval_calls", 0),
        "image_retrieval_calls": agent_metrics.get("image_retrieval_calls", 0),
        "hallucination_free": judge_scores["hallucination_free"],
        "claim_support_consistency": judge_scores["claim_support_consistency"],
    }


# ---------------------------------------------------------------------------
# 기능 E: 메인 배치 실행
# ---------------------------------------------------------------------------

def run_cikm_experiment():
    print("=" * 65)
    print("  PRISM CIKM Batch Evaluation 시작 (Resume 지원)")
    print("  샘플: sampled_300_v2.csv (seed=42, Supported 150 + Refuted 150)")
    print("=" * 65)

    # 1. 검증 완료 canonical 샘플 로드 (추가 샘플링/셔플 없음)
    samples = _load_samples()
    lbl_cnt = Counter(s["label"] for s in samples)
    total   = len(samples)
    print(f"   ✅ 로드 완료: 총 {total}개")
    print(f"      supported={lbl_cnt.get('supported', 0)}, refuted={lbl_cnt.get('refuted', 0)}")

    # 2. Resume: cikm_final_metrics.csv 가 있으면 이미 처리된 claim_id 파악
    done_ids: set[str] = set()
    write_header = True   # 파일 첫 기록 시에만 헤더를 씀

    if os.path.exists(OUTPUT_FINAL):
        try:
            prev_df = pd.read_csv(OUTPUT_FINAL)
            if "claim_id" in prev_df.columns and len(prev_df) > 0:
                done_ids    = set(prev_df["claim_id"].astype(str))
                write_header = False
                print(f"   ♻️  이전 결과 {len(done_ids)}개 감지 "
                      f"— {len(done_ids) + 1}번째 클레임부터 이어서 실행합니다.\n")
            else:
                print("   ⚠️  출력 파일이 비어 있거나 claim_id 컬럼 없음 "
                      "— 처음부터 시작합니다.\n")
        except Exception as e:
            print(f"   ⚠️  출력 파일 로드 실패({e}) — 처음부터 시작합니다.\n")
    else:
        print()

    processed = 0

    for idx, sample in enumerate(samples, start=1):
        cid = sample["claim_id"]

        if cid in done_ids:
            print(f"[{idx:03d}/{total}] ⏩ 건너뜀 (이미 완료): {cid}")
            continue

        claim = sample["claim"]
        print(f"\n[{idx:03d}/{total}] 클레임: {claim[:70]}...")

        row        = None
        e2e_start  = time.time()

        # ── 에러 시 1회 재시도 → 재시도도 실패하면 안전 종료 ──────────────────
        for attempt in range(1, 3):
            try:
                row = _process_one(idx, sample, e2e_start)
                break  # 성공
            except Exception as e:
                print(f"   [실행 에러 - 시도 {attempt}/2] {type(e).__name__}: {e}")
                if attempt < 2:
                    print("   ⏳ 10초 후 재시도합니다...")
                    time.sleep(10)
                    e2e_start = time.time()   # 재시도 시 타이머 리셋
                else:
                    # 2회 모두 실패 → 현재까지 저장된 데이터 보존 후 종료
                    print("\n" + "=" * 65)
                    print("  ❌ 재시도 실패. 현재까지의 결과는 안전하게 저장되었습니다.")
                    print(f"     다시 실행하면 [{idx:03d}]번 클레임부터 이어서 시작합니다.")
                    print(f"     완료: {processed}개 / 남은 클레임: {total - len(done_ids) - processed}개")
                    print("=" * 65)
                    sys.exit(1)

        # ── 클레임 1개 완료 → 즉시 append 저장 ────────────────────────────────
        row_df = pd.DataFrame([row], columns=OUTPUT_COLUMNS)
        row_df.to_csv(
            OUTPUT_FINAL,
            mode="a",
            header=write_header,
            index=False,
            encoding="utf-8-sig",
        )
        write_header = False   # 이후 행은 헤더 없이 append
        processed += 1
        print(f"   💾 저장 완료 ({idx}/{total}) — pred_label: {row['pred_label']}")

    # ---------------------------------------------------------------------------
    # 최종 요약 통계 출력
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * 65}")
    print(f"  ✅ 전체 처리 완료! 신규 처리: {processed}개 / 누적: {total}개")
    print(f"  출력 파일: {OUTPUT_FINAL}")
    print("=" * 65)

    df    = pd.read_csv(OUTPUT_FINAL)
    valid = df.dropna(subset=["total_tokens", "e2e_latency_sec"])

    print(f"\n  평가 완료 샘플 수         : {len(valid)} / {len(df)}")
    print(f"  총 토큰 합계 (sum)        : {valid['total_tokens'].sum():,.0f}")
    print(f"  평균 토큰 / 샘플          : {valid['total_tokens'].mean():,.1f}")
    print(f"  총 추정 비용 (USD)        : ${valid['total_cost_usd'].sum():.4f}")

    lat = valid["e2e_latency_sec"]
    print(f"\n  E2E Latency (sec)")
    print(f"    Median                 : {lat.median():.2f}")
    print(f"    Mean                   : {lat.mean():.2f}")
    print(f"    P90                    : {np.percentile(lat, 90):.2f}")
    print(f"    P95                    : {np.percentile(lat, 95):.2f}")

    print(f"\n  Retrieval 호출 횟수 (평균)")
    print(f"    text_retrieval_calls   : {valid['text_retrieval_calls'].mean():.2f}")
    print(f"    image_retrieval_calls  : {valid['image_retrieval_calls'].mean():.2f}")

    print(f"\n  Explanation Quality (LLM-as-a-Judge)")
    print(f"    hallucination_free     : {valid['hallucination_free'].mean():.3f}")
    print(f"    claim_support_consist. : {valid['claim_support_consistency'].mean():.3f}")

    if "pred_label" in df.columns:
        print(f"\n  Prediction Distribution")
        for lbl, cnt in df["pred_label"].value_counts().items():
            print(f"    {lbl:<20}: {cnt}")

    print("=" * 65)

    return df


if __name__ == "__main__":
    run_cikm_experiment()
