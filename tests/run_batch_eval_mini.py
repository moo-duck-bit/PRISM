"""
run_batch_eval_mini.py — 50개 디버그 미니 테스트 (기존 run_batch_eval.py와 완전 분리)

[기존 run_batch_eval.py 와의 차이점]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
① 샘플: Supported 앞에서 25개 + Refuted 앞에서 25개 = 총 50개
② Resume 로직: 비활성화 (무조건 1번부터 재실행)
③ 출력 파일: cikm_debug_metrics.csv (본 실험 덮어쓰기 방지)
④ 완료 후 핵심 디버깅 수치 3가지 즉시 출력:
   - 총 NEI 발생 횟수 (목표: 5개 미만)
   - Supported 25개 중 모델이 Supported로 맞춘 수 (Recall)
   - Supported 25개의 평균 Hallucination-Free 점수
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

검증 목표:
  - NEI < 5개 → anti-NEI 프롬프트 작동 확인
  - Supported Recall ≥ 20/25 → 판정 방향성 확인
  - Hallucination-Free ≥ 0.70 → Toulmin 안전장치 확인
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

# ── Aggressive 파이프라인 사용 (4개 verifier 모두 anti-NEI + generator_toulmin 반영) ──
# 근거: entity/spatiotemporal(spaCy 없음)·visual(이미지 없음) verifier가
#       구조적으로 항상 NEI를 반환 → 원본 다수결에서 NEI가 2~3표 고정이므로
#       NEI 기권 처리 집계 + aggressive verifier 조합이 필수
from prism.core.graph_aggressive import prism_app_aggressive as prism_app
from prism.agents.triage import triage_agent

# ---------------------------------------------------------------------------
# 상수 설정
# ---------------------------------------------------------------------------
SAMPLED_CSV  = "/home/jkpark/prism/sampled_300_v2.csv"
CORPUS2_CSV  = "/home/jkpark/prism/mocheg_json_csv/mocheg/test/Corpus2.csv"

OUTPUT_FINAL = "cikm_debug_metrics.csv"   # ← 본 실험 파일과 완전 분리
JUDGE_MODEL  = "gpt-4o"

MINI_N_EACH  = 25    # Supported 25 + Refuted 25 = 50개

OUTPUT_COLUMNS = [
    "idx", "claim_id", "claim", "ground_truth", "pred_label",
    "total_tokens", "prompt_tokens", "completion_tokens", "total_cost_usd",
    "e2e_latency_sec",
    "triage_latency", "retrieval_text_latency", "retrieval_visual_latency",
    "factual_verifier_latency", "generator_latency",
    "text_retrieval_calls", "image_retrieval_calls",
    "hallucination_free", "claim_support_consistency",
    "verdict_factual", "verdict_entity", "verdict_spatiotemporal", "verdict_visual",
]


# ---------------------------------------------------------------------------
# 집계 함수 — Aggressive (NEI 기권 처리)
# ---------------------------------------------------------------------------
# 원본 다수결과의 차이:
#   - NEI 투표는 '기권'으로 처리, 비-NEI 투표만으로 다수결
#   - 동수(tie) 시 Supported 우선 (기존: NEI 우선)
#   - 모든 verifier가 NEI일 때만 최종 NEI 반환
# 적용 이유: spaCy 미설치·이미지 부재로 entity/spatiotemporal/visual이
#           구조적으로 NEI를 반환 → 원본 집계 시 NEI 압도가 불가피

def _aggregate_verdict(verdicts: dict) -> str:
    all_votes = [v.get("verdict", "NEI") for v in verdicts.values() if isinstance(v, dict)]
    if not all_votes:
        return "NEI"
    decisive = [v for v in all_votes if v != "NEI"]
    if not decisive:
        return "NEI"
    cnt = Counter(decisive)
    top_label, top_count = cnt.most_common(1)[0]
    if list(cnt.values()).count(top_count) > 1:
        return "Supported" if "Supported" in cnt else "Refuted"
    return top_label


# ---------------------------------------------------------------------------
# LLM-as-a-Judge
# ---------------------------------------------------------------------------
_judge_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0)


def evaluate_explanation_quality(claim: str, explanation: str, ground_truth: str) -> dict:
    prompt = f"""You are an expert fact-checking evaluator. Assess the following explanation strictly.

Claim: {claim}
Ground Truth Label: {ground_truth}
Generated Explanation: {explanation}

Evaluate on TWO criteria and respond ONLY in valid JSON:
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
# 미니 샘플 로드: Supported 25 + Refuted 25
# ---------------------------------------------------------------------------

def _load_mini_samples() -> list[dict]:
    """
    sampled_300_v2.csv에서 label 기준으로 앞에서부터
    Supported(=supported) 25개 + Refuted(=refuted) 25개 = 총 50개 추출.
    label은 대소문자 무관하게 처리합니다.
    """
    if not os.path.exists(CORPUS2_CSV):
        raise FileNotFoundError(f"Corpus2.csv를 찾을 수 없습니다: {CORPUS2_CSV}")

    claim_text: dict[str, str] = {}
    with open(CORPUS2_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row["claim_id"].strip()
            if cid not in claim_text:
                claim_text[cid] = row["Claim"].strip()

    if not os.path.exists(SAMPLED_CSV):
        raise FileNotFoundError(f"sampled_300_v2.csv를 찾을 수 없습니다: {SAMPLED_CSV}")

    supported_samples: list[dict] = []
    refuted_samples:   list[dict] = []

    with open(SAMPLED_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row["claim_id"].strip()
            lbl = row["label"].strip().lower()
            text = claim_text.get(cid, "")
            if not text:
                continue
            entry = {"claim_id": cid, "claim": text, "label": lbl}
            if lbl == "supported" and len(supported_samples) < MINI_N_EACH:
                supported_samples.append(entry)
            elif lbl == "refuted" and len(refuted_samples) < MINI_N_EACH:
                refuted_samples.append(entry)
            if len(supported_samples) >= MINI_N_EACH and len(refuted_samples) >= MINI_N_EACH:
                break

    samples = supported_samples + refuted_samples
    print(f"   [미니 샘플] supported={len(supported_samples)}, refuted={len(refuted_samples)}, 합계={len(samples)}")
    return samples


# ---------------------------------------------------------------------------
# 단일 클레임 처리
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
    final_verdict = _aggregate_verdict(verdicts)

    verdict_factual        = verdicts.get("factual",        {}).get("verdict", "N/A")
    verdict_entity         = verdicts.get("entity",         {}).get("verdict", "N/A")
    verdict_spatiotemporal = verdicts.get("spatiotemporal", {}).get("verdict", "N/A")
    verdict_visual         = verdicts.get("visual",         {}).get("verdict", "N/A")

    print(f"   📊 factual={verdict_factual} | entity={verdict_entity} | "
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
# 메인 실행
# ---------------------------------------------------------------------------

def run_mini_experiment():
    print("=" * 65)
    print("  PRISM 미니 디버그 테스트 (50개 — Supported 25 + Refuted 25)")
    print("  출력 파일: cikm_debug_metrics.csv")
    print("  ⚠️  Resume 비활성화 — 무조건 처음부터 새로 실행")
    print("=" * 65)

    # 미니 샘플 로드
    samples = _load_mini_samples()
    total   = len(samples)

    # ── Resume 로직 비활성화: 기존 파일 무시하고 항상 새로 시작 ──────────────
    # (run_batch_eval.py 의 Resume 블록은 의도적으로 제거)
    write_header = True
    processed = 0

    for idx, sample in enumerate(samples, start=1):
        label_tag = f"[{'S' if sample['label'] == 'supported' else 'R'}{idx:02d}]"
        claim = sample["claim"]
        print(f"\n{label_tag} [{idx:02d}/{total}] {claim[:65]}...")

        row       = None
        e2e_start = time.time()

        for attempt in range(1, 3):
            try:
                row = _process_one(idx, sample, e2e_start)
                break
            except Exception as e:
                print(f"   [에러 - 시도 {attempt}/2] {type(e).__name__}: {e}")
                if attempt < 2:
                    print("   ⏳ 10초 후 재시도...")
                    time.sleep(10)
                    e2e_start = time.time()
                else:
                    print(f"   ❌ 재시도 실패 — [{idx:02d}]번 스킵하고 계속 진행합니다.")
                    row = {col: None for col in OUTPUT_COLUMNS}
                    row.update({"idx": idx, "claim_id": sample["claim_id"],
                                "claim": claim, "ground_truth": sample["label"],
                                "pred_label": "ERROR"})

        row_df = pd.DataFrame([row], columns=OUTPUT_COLUMNS)
        row_df.to_csv(
            OUTPUT_FINAL,
            mode="w" if write_header else "a",
            header=write_header,
            index=False,
            encoding="utf-8-sig",
        )
        write_header = False
        processed += 1
        print(f"   💾 저장 ({idx}/{total}) — pred={row['pred_label']} | gt={row['ground_truth']}")

    # ── 기본 요약 통계 ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  ✅ 미니 테스트 완료! 처리: {processed}개")
    print("=" * 65)

    df    = pd.read_csv(OUTPUT_FINAL)
    valid = df.dropna(subset=["total_tokens", "e2e_latency_sec"])

    if len(valid) > 0:
        print(f"\n  평가 완료 샘플 수         : {len(valid)} / {len(df)}")
        print(f"  총 토큰 합계              : {valid['total_tokens'].sum():,.0f}")
        print(f"  총 추정 비용 (USD)        : ${valid['total_cost_usd'].sum():.4f}")
        lat = valid["e2e_latency_sec"]
        print(f"  E2E Latency 평균 (sec)   : {lat.mean():.2f}")

        print(f"\n  Prediction Distribution")
        for lbl, cnt in df["pred_label"].value_counts().items():
            pct = cnt / len(df) * 100
            print(f"    {lbl:<20}: {cnt:>3}  ({pct:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════════
    # ★ 핵심 디버깅 수치 3가지 (판단 기준 명시)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'━' * 65}")
    print("  ★ 핵심 디버깅 수치 (Anti-NEI + Anti-Hallucination 검증)")
    print(f"{'━' * 65}")

    # 1) 총 NEI 발생 횟수
    nei_count = (df["pred_label"] == "NEI").sum()
    nei_status = "✅ PASS (목표 달성)" if nei_count < 5 else f"❌ FAIL (목표: 5개 미만)"
    print(f"\n  [1] 총 NEI 발생 횟수      : {nei_count}개  {nei_status}")

    # 2) Supported 25개 중 모델이 Supported로 맞춘 수 (Recall)
    sup_df   = df[df["ground_truth"].str.lower() == "supported"]
    sup_total = len(sup_df)
    sup_correct = (sup_df["pred_label"].str.lower() == "supported").sum()
    sup_recall  = sup_correct / sup_total if sup_total > 0 else 0.0
    recall_status = "✅ PASS" if sup_correct >= 20 else "❌ FAIL (목표: 20개 이상)"
    print(f"  [2] Supported Recall      : {sup_correct}/{sup_total}  "
          f"({sup_recall:.1%})  {recall_status}")

    # 3) Supported 25개의 평균 Hallucination-Free 점수
    sup_hf = sup_df["hallucination_free"].dropna()
    sup_hf_mean = sup_hf.mean() if len(sup_hf) > 0 else 0.0
    hf_status = "✅ PASS" if sup_hf_mean >= 0.70 else "❌ FAIL (목표: 0.70 이상)"
    print(f"  [3] Supported HalluFree   : {sup_hf_mean:.3f}  {hf_status}")

    print(f"{'━' * 65}\n")

    return df


if __name__ == "__main__":
    run_mini_experiment()
