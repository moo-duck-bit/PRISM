"""
run_batch_eval_aggressive.py — Anti-NEI 강화 버전 배치 평가 스크립트

[기존 run_batch_eval.py 와의 차이점]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
① 파이프라인: prism_app → prism_app_aggressive (graph_aggressive.py)
   - 4개 Verifier 전부 Anti-NEI 강화 프롬프트 버전으로 교체

② 집계 함수: _aggregate_verdict → _aggregate_verdict_aggressive
   - 기존: 동수(tie) → NEI 우선
   - 변경: 동수(tie) → NEI를 제외하고 재투표, 그래도 동수면 Supported 우선
   - 이유: NEI 기권 표를 의사결정에서 배제하여 Recall 개선

③ 출력 파일: cikm_final_metrics.csv → cikm_final_metrics_aggressive.csv
   (기존 결과 보존, 비교 가능)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

비교 방법:
  - 기존 버전 결과: cikm_final_metrics.csv
  - Aggressive 버전: cikm_final_metrics_aggressive.csv
  - 지표 비교 스크립트: compare_experiments.py (별도 제공)

[샘플링 정책 — 기존과 동일]
- 소스: sampled_300_v2.csv (seed=42, NEI 제외, Supported 150 + Refuted 150)
- 재현성: verify_reproducibility_v2.py 로 PASS 확인된 canonical 목록

[Resume 전략]
- cikm_final_metrics_aggressive.csv 가 존재하면 이미 처리된 claim_id 건너뜀
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# ── Aggressive 파이프라인 임포트 (기존 prism_app 과 분리) ─────────────────────
from prism.core.graph_aggressive import prism_app_aggressive
from prism.agents.triage import triage_agent

# ---------------------------------------------------------------------------
# 상수 설정
# ---------------------------------------------------------------------------
SAMPLED_CSV  = "/home/jkpark/prism/sampled_300_v2.csv"
CORPUS2_CSV  = "/home/jkpark/prism/mocheg_json_csv/mocheg/test/Corpus2.csv"

# 기존과 다른 출력 파일 → 두 결과를 side-by-side 비교 가능
OUTPUT_FINAL = "cikm_final_metrics_aggressive.csv"
JUDGE_MODEL  = "gpt-4o"

OUTPUT_COLUMNS = [
    "idx", "claim_id", "claim", "ground_truth", "pred_label",
    "total_tokens", "prompt_tokens", "completion_tokens", "total_cost_usd",
    "e2e_latency_sec",
    "triage_latency", "retrieval_text_latency", "retrieval_visual_latency",
    "factual_verifier_latency", "generator_latency",
    "text_retrieval_calls", "image_retrieval_calls",
    "hallucination_free", "claim_support_consistency",
    # Aggressive 버전 추가 컬럼: 각 verifier별 개별 verdict 저장 (디버깅용)
    "verdict_factual", "verdict_entity", "verdict_spatiotemporal", "verdict_visual",
]


# ---------------------------------------------------------------------------
# 기능 A: 최종 판정 집계 (Aggressive — NEI 기권 제거 후 재투표)
# ---------------------------------------------------------------------------

def _aggregate_verdict_aggressive(verdicts: dict) -> str:
    """
    [Aggressive 버전] Anti-NEI 집계 함수.

    변경 사항 (기존 _aggregate_verdict 대비):
    1. NEI 투표는 '기권'으로 처리하고 비-NEI 투표만으로 다수결 진행
    2. 비-NEI 투표도 동수(tie)면 "Supported" 우선 (기존: NEI 우선)
    3. 모든 투표가 NEI인 경우에만 최종 NEI 반환

    반환: "Supported" | "Refuted" | "NEI"
    """
    all_votes = [v.get("verdict", "NEI") for v in verdicts.values() if isinstance(v, dict)]

    if not all_votes:
        return "NEI"

    # ── Step 1: NEI를 제외한 실질 투표만 집계 ──────────────────────────────
    decisive_votes = [v for v in all_votes if v != "NEI"]

    if not decisive_votes:
        # 모든 verifier가 NEI를 반환한 경우만 최종 NEI
        return "NEI"

    cnt = Counter(decisive_votes)
    top_label, top_count = cnt.most_common(1)[0]

    # ── Step 2: 동수(tie) 처리 → Supported 우선 (기존: NEI 우선) ──────────
    if list(cnt.values()).count(top_count) > 1:
        # 동수면 Supported 우선 (NEI 대신)
        return "Supported" if "Supported" in cnt else "Refuted"

    return top_label


# ---------------------------------------------------------------------------
# 기능 B: LLM-as-a-Judge (기존과 동일)
# ---------------------------------------------------------------------------
_judge_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0)


def evaluate_explanation_quality(claim: str, explanation: str, ground_truth: str) -> dict:
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
# 기능 C: 데이터 로드 (기존과 동일)
# ---------------------------------------------------------------------------

def _load_samples() -> list[dict]:
    if not os.path.exists(CORPUS2_CSV):
        raise FileNotFoundError(f"Corpus2.csv를 찾을 수 없습니다: {CORPUS2_CSV}")

    claim_text: dict[str, str] = {}
    with open(CORPUS2_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row["claim_id"].strip()
            if cid not in claim_text:
                claim_text[cid] = row["Claim"].strip()

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

    return samples


# ---------------------------------------------------------------------------
# 기능 D: 단일 클레임 처리
# ---------------------------------------------------------------------------

def _process_one(idx: int, sample: dict, e2e_start: float) -> dict:
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
        # ── Aggressive 파이프라인 호출 (기존: prism_app.invoke) ──────────────
        final_state = prism_app_aggressive.invoke(initial_state)

    e2e_latency   = round(time.time() - e2e_start, 4)
    agent_metrics = final_state.get("metrics", {})
    final_article = final_state.get("final_article", "")
    verdicts      = final_state.get("verdicts", {})

    # ── Aggressive 집계 함수 호출 (기존: _aggregate_verdict) ─────────────────
    final_verdict = _aggregate_verdict_aggressive(verdicts)

    # 각 verifier 개별 verdict 기록 (디버깅 및 분석용)
    verdict_factual          = verdicts.get("factual",         {}).get("verdict", "N/A")
    verdict_entity           = verdicts.get("entity",          {}).get("verdict", "N/A")
    verdict_spatiotemporal   = verdicts.get("spatiotemporal",  {}).get("verdict", "N/A")
    verdict_visual           = verdicts.get("visual",          {}).get("verdict", "N/A")

    print(f"   📊 [개별 판정] factual={verdict_factual} | entity={verdict_entity} | "
          f"spatio={verdict_spatiotemporal} | visual={verdict_visual} → 최종={final_verdict}")

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
        "verdict_factual": verdict_factual,
        "verdict_entity": verdict_entity,
        "verdict_spatiotemporal": verdict_spatiotemporal,
        "verdict_visual": verdict_visual,
    }


# ---------------------------------------------------------------------------
# 기능 E: 메인 배치 실행
# ---------------------------------------------------------------------------

def run_cikm_experiment_aggressive():
    print("=" * 70)
    print("  PRISM CIKM Batch Evaluation — AGGRESSIVE (Anti-NEI) 버전")
    print("  출력: cikm_final_metrics_aggressive.csv")
    print("  비교: cikm_final_metrics.csv (기존 보수적 버전)")
    print("=" * 70)

    samples = _load_samples()
    lbl_cnt = Counter(s["label"] for s in samples)
    total   = len(samples)
    print(f"   ✅ 로드 완료: 총 {total}개")
    print(f"      supported={lbl_cnt.get('supported', 0)}, refuted={lbl_cnt.get('refuted', 0)}")

    done_ids: set[str] = set()
    write_header = True

    if os.path.exists(OUTPUT_FINAL):
        try:
            prev_df = pd.read_csv(OUTPUT_FINAL)
            if "claim_id" in prev_df.columns and len(prev_df) > 0:
                done_ids    = set(prev_df["claim_id"].astype(str))
                write_header = False
                print(f"   ♻️  이전 결과 {len(done_ids)}개 감지 "
                      f"— {len(done_ids) + 1}번째 클레임부터 이어서 실행합니다.\n")
            else:
                print("   ⚠️  출력 파일이 비어 있거나 claim_id 컬럼 없음 — 처음부터 시작합니다.\n")
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

        row       = None
        e2e_start = time.time()

        for attempt in range(1, 3):
            try:
                row = _process_one(idx, sample, e2e_start)
                break
            except Exception as e:
                print(f"   [실행 에러 - 시도 {attempt}/2] {type(e).__name__}: {e}")
                if attempt < 2:
                    print("   ⏳ 10초 후 재시도합니다...")
                    time.sleep(10)
                    e2e_start = time.time()
                else:
                    print("\n" + "=" * 70)
                    print("  ❌ 재시도 실패. 현재까지의 결과는 안전하게 저장되었습니다.")
                    print(f"     다시 실행하면 [{idx:03d}]번 클레임부터 이어서 시작합니다.")
                    print(f"     완료: {processed}개 / 남은 클레임: {total - len(done_ids) - processed}개")
                    print("=" * 70)
                    sys.exit(1)

        row_df = pd.DataFrame([row], columns=OUTPUT_COLUMNS)
        row_df.to_csv(
            OUTPUT_FINAL,
            mode="a",
            header=write_header,
            index=False,
            encoding="utf-8-sig",
        )
        write_header = False
        processed += 1
        print(f"   💾 저장 완료 ({idx}/{total}) — pred_label: {row['pred_label']}")

    # ── 최종 요약 통계 ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  ✅ [AGGRESSIVE] 전체 처리 완료! 신규 처리: {processed}개 / 누적: {total}개")
    print(f"  출력 파일: {OUTPUT_FINAL}")
    print("=" * 70)

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
        print(f"\n  Prediction Distribution (Aggressive)")
        for lbl, cnt in df["pred_label"].value_counts().items():
            pct = cnt / len(df) * 100
            print(f"    {lbl:<20}: {cnt:>4}  ({pct:.1f}%)")

    # ── 기존 결과와 비교 (파일이 있는 경우) ──────────────────────────────────
    baseline_csv = os.path.join(os.path.dirname(OUTPUT_FINAL), "cikm_final_metrics.csv")
    if os.path.exists(baseline_csv):
        print(f"\n{'─' * 70}")
        print("  📊 기존 버전 vs Aggressive 버전 비교")
        print(f"{'─' * 70}")
        baseline_df = pd.read_csv(baseline_csv)
        for lbl in ["Supported", "Refuted", "NEI", "supported", "refuted"]:
            b_cnt = (baseline_df["pred_label"] == lbl).sum()
            a_cnt = (df["pred_label"] == lbl).sum()
            if b_cnt > 0 or a_cnt > 0:
                print(f"    {lbl:<20}: 기존={b_cnt:>4}  →  Aggressive={a_cnt:>4}")
        print(f"{'─' * 70}")
    else:
        print(f"\n  ℹ️  기존 결과(cikm_final_metrics.csv)가 없어 비교를 건너뜁니다.")

    print("=" * 70)

    return df


if __name__ == "__main__":
    run_cikm_experiment_aggressive()
