"""배치 평가 실행기.

한 건을 끝낼 때마다 CSV에 append 한다. 장시간 실행 중 끊겨도 결과가 남고,
같은 명령을 다시 실행하면 이미 처리한 claim_id를 건너뛰고 이어서 돈다.

사용 예:
    # MOCHEG 균형 이진 300
    python -m prism.evaluation.batch_eval --dataset mocheg --setting binary \
        --n-per-class 150 --output results/mocheg_bin.csv

    # MOCHEG 3-class 전체
    python -m prism.evaluation.batch_eval --dataset mocheg --setting three_class \
        --output results/mocheg_3cls.csv

    # RW-Post, 증거 게이팅 제거 ablation
    python -m prism.evaluation.batch_eval --dataset rwpost --data data/rwpost.json \
        --variant no_gating --output results/rwpost_closed.csv

    # 베이스라인
    python -m prism.evaluation.batch_eval --dataset mocheg --baseline zero_shot \
        --output results/b2_zeroshot.csv
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import pandas as pd
from langchain_community.callbacks import get_openai_callback

from prism.core.graph import ABLATIONS, build_variant, initial_state
from prism.core.verdict import NEI, TEXT_VERIFIERS, VISUAL_VERIFIER
from prism.evaluation import judge, metrics
from prism.evaluation.baselines import BASELINES
from prism.evaluation.dataset import Sample, load_mocheg, load_rwpost

logger = logging.getLogger(__name__)

RETRY_LIMIT = 2
RETRY_WAIT_SEC = 10

VERIFIERS = TEXT_VERIFIERS + (VISUAL_VERIFIER,)

OUTPUT_COLUMNS = [
    "idx", "claim_id", "claim", "gold", "pred",
    # triage
    "mask_img", "mask_ent", "mask_time", "active_agent_count",
    # 검증기 판정
    *[f"verdict_{name}" for name in VERIFIERS],
    *[f"confidence_{name}" for name in VERIFIERS],
    # 캘리브레이션
    "calib_path", "calib_rule", "calib_conflict",
    # 비용과 지연
    "total_tokens", "prompt_tokens", "completion_tokens", "total_cost_usd",
    "e2e_latency_sec", "triage_latency", "retrieval_text_latency", "generator_latency",
    "n_text_evidence", "n_image_evidence",
    # 설명 품질
    "structural_completeness", "faithfulness", "n_propositions", "decision_fidelity",
]


def _row_from_state(idx: int, sample: Sample, state: Dict, usage, latency: float,
                    scores: Dict) -> Dict:
    verdicts = state.get("verdicts", {})
    mask = state.get("modality_mask", {})
    calibration = state.get("calibration", {})
    agent_metrics = state.get("metrics", {})

    row = {
        "idx": idx,
        "claim_id": sample.claim_id,
        "claim": sample.claim,
        "gold": sample.label,
        "pred": state.get("verdict", NEI),
        "mask_img": int(bool(mask.get("img"))),
        "mask_ent": int(bool(mask.get("ent"))),
        "mask_time": int(bool(mask.get("time"))),
        "active_agent_count": agent_metrics.get("active_agent_count"),
        "calib_path": calibration.get("path"),
        "calib_rule": calibration.get("rule"),
        "calib_conflict": int(bool(calibration.get("conflict"))),
        "total_tokens": usage.total_tokens,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_cost_usd": round(usage.total_cost, 6),
        "e2e_latency_sec": latency,
        "n_text_evidence": len(state.get("text_evidence", [])),
        "n_image_evidence": len(state.get("image_evidence", [])),
        "structural_completeness": agent_metrics.get("structural_completeness"),
    }
    for key in ("triage_latency", "retrieval_text_latency", "generator_latency"):
        row[key] = agent_metrics.get(key)
    for name in VERIFIERS:
        result = verdicts.get(name) or {}
        row[f"verdict_{name}"] = result.get("verdict")
        row[f"confidence_{name}"] = result.get("confidence")
    row.update(scores)
    return row


def _process_one(idx: int, sample: Sample, app, started: float, score_explanation: bool) -> Dict:
    state = initial_state(
        claim=sample.claim,
        claim_id=sample.claim_id,
        post_text=sample.post_text,
        ocr_text=sample.ocr_text,
        image_paths=sample.image_paths,
        evidence_pool=sample.evidence_pool,
    )

    with get_openai_callback() as usage:
        final_state = app.invoke(state)

        scores = {"faithfulness": None, "n_propositions": None, "decision_fidelity": None}
        if score_explanation and final_state.get("report"):
            rationales = [
                r.get("rationale", "")
                for r in final_state.get("verdicts", {}).values()
                if isinstance(r, dict)
            ]
            scores = judge.score_explanation(
                explanation=final_state["report"],
                verdict=final_state.get("verdict", NEI),
                text_evidence=final_state.get("text_evidence", []),
                rationales=rationales,
                has_image=bool(final_state.get("image_evidence")),
            )

    return _row_from_state(idx, sample, final_state, usage,
                           round(time.time() - started, 4), scores)


def _completed_ids(output_path: str) -> set:
    if not os.path.exists(output_path):
        return set()

    existing = pd.read_csv(output_path)
    if existing.empty:
        return set()

    if set(existing.columns) != set(OUTPUT_COLUMNS):
        raise SystemExit(
            f"기존 파일 {output_path} 의 컬럼 구성이 현재 스크립트와 다릅니다. "
            "다른 --output 경로를 쓰거나 --no-resume 으로 새로 시작하세요."
        )
    return set(existing["claim_id"].astype(str))


def _append(row: Dict, output_path: str, write_header: bool) -> None:
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    pd.DataFrame([row], columns=OUTPUT_COLUMNS).to_csv(
        output_path,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
        encoding="utf-8-sig",
    )


def summarize(output_path: str) -> pd.DataFrame:
    """누적 결과에서 판정 지표와 비용을 출력한다."""
    df = pd.read_csv(output_path)
    if df.empty:
        print("결과가 비어 있습니다.")
        return df

    scores = metrics.evaluate(df["gold"], df["pred"])
    lower, upper = metrics.bootstrap_ci(df["gold"], df["pred"], "accuracy")

    print(f"\n결과 파일: {output_path}")
    print(metrics.format_report(scores))
    print(f"Acc 95% CI [{lower:.3f}, {upper:.3f}]")

    valid = df.dropna(subset=["total_tokens"])
    if not valid.empty:
        print(f"\n토큰/건 {valid['total_tokens'].mean():,.0f}   "
              f"비용/건 ${valid['total_cost_usd'].mean():.5f}   "
              f"총 ${valid['total_cost_usd'].sum():.4f}")
        print(f"E2E latency 중앙값 {valid['e2e_latency_sec'].median():.2f}s")

    if valid["active_agent_count"].notna().any():
        print(f"활성 에이전트 평균 {valid['active_agent_count'].mean():.2f}/4")
        for name in VERIFIERS:
            column = f"verdict_{name}"
            rate = df[column].notna().mean()
            print(f"  {name:<16} 활성률 {rate:.1%}")

    if df["faithfulness"].notna().any():
        print(f"\nFaithfulness {df['faithfulness'].mean():.3f}   "
              f"Decision-Fidelity {df['decision_fidelity'].mean():.3f}   "
              f"구조 완전성 {df['structural_completeness'].mean():.3f}")

    if df["calib_rule"].notna().any():
        print("\n캘리브레이션 규칙 분포")
        for rule, count in df["calib_rule"].value_counts().items():
            print(f"  {rule:<22} {count}")
        print(f"  교차모달 충돌 발생 {int(df['calib_conflict'].sum())}건")

    return df


def load_dataset(args: argparse.Namespace) -> List[Sample]:
    if args.dataset == "mocheg":
        samples = load_mocheg(
            labels_csv=args.labels,
            claims_csv=args.claims,
            setting=args.setting,
            n_per_class=args.n_per_class,
        )
    else:
        samples = load_rwpost(path=args.data, image_root=args.image_root)

    if args.limit:
        samples = samples[: args.limit]
    return samples


def run_baseline(name: str, samples: Sequence[Sample], output_path: str) -> pd.DataFrame:
    """베이스라인은 파이프라인을 타지 않으므로 별도 경로로 처리한다."""
    predictions = BASELINES[name](samples)
    rows = []
    for idx, (sample, pred) in enumerate(zip(samples, predictions), start=1):
        row = {column: None for column in OUTPUT_COLUMNS}
        row.update(idx=idx, claim_id=sample.claim_id, claim=sample.claim,
                   gold=sample.label, pred=pred)
        rows.append(row)

    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(
        output_path, index=False, encoding="utf-8-sig"
    )
    return summarize(output_path)


def run(samples: Sequence[Sample], output_path: str, variant: str,
        score_explanation: bool, generate_explanation: bool, resume: bool) -> pd.DataFrame:
    done = _completed_ids(output_path) if resume else set()
    write_header = not done
    if done:
        logger.info("이미 처리한 %d건을 건너뜁니다", len(done))

    app = build_variant(variant, generate_explanation=generate_explanation)
    total = len(samples)
    processed = 0

    for idx, sample in enumerate(samples, start=1):
        if sample.claim_id in done:
            continue

        print(f"\n[{idx:04d}/{total}] {sample.claim[:70]}...")
        row = None
        started = time.time()

        for attempt in range(1, RETRY_LIMIT + 1):
            try:
                row = _process_one(idx, sample, app, started, score_explanation)
                break
            except Exception as exc:
                logger.error("처리 실패 (시도 %d/%d): %s: %s",
                             attempt, RETRY_LIMIT, type(exc).__name__, exc)
                if attempt < RETRY_LIMIT:
                    time.sleep(RETRY_WAIT_SEC)
                    started = time.time()

        if row is None:
            print(f"\n{idx}번에서 중단했습니다. 여기까지의 결과는 저장되어 있습니다.")
            print(f"신규 처리 {processed}건 / 남은 {total - len(done) - processed}건")
            sys.exit(1)

        _append(row, output_path, write_header)
        write_header = False
        processed += 1
        print(f"  pred={row['pred']} gold={row['gold']} "
              f"rule={row['calib_rule']} agents={row['active_agent_count']}")

    print(f"\n신규 처리 {processed}건 완료")
    return summarize(output_path)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PRISM 배치 평가")

    parser.add_argument("--dataset", choices=("mocheg", "rwpost"), default="mocheg")
    parser.add_argument("--output", required=True, help="결과를 append 할 CSV 경로")

    mocheg = parser.add_argument_group("mocheg")
    mocheg.add_argument("--labels", default=os.getenv("PRISM_LABELS_CSV", "data/politifact_labels.csv"))
    mocheg.add_argument("--claims", default=os.getenv("PRISM_CLAIMS_CSV", "data/Corpus2.csv"))
    mocheg.add_argument("--setting", choices=("binary", "three_class"), default="three_class")
    mocheg.add_argument("--n-per-class", type=int, default=None,
                        help="클래스별 표본 수. binary 150이면 균형 300 세트")

    rwpost = parser.add_argument_group("rwpost")
    rwpost.add_argument("--data", default=os.getenv("PRISM_RWPOST_PATH", "data/rwpost.json"))
    rwpost.add_argument("--image-root", default=os.getenv("PRISM_IMAGE_ROOT", ""))

    parser.add_argument("--variant", choices=tuple(ABLATIONS), default="full")
    parser.add_argument("--baseline", choices=tuple(BASELINES), default=None,
                        help="지정하면 PRISM 대신 해당 베이스라인을 실행한다")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-explanation", action="store_true",
                        help="Toulmin 생성 노드를 빼고 판정만 측정 (비용 절감)")
    parser.add_argument("--no-judge", action="store_true",
                        help="설명 품질 자동 평가를 건너뛴다")
    parser.add_argument("--no-resume", action="store_true")

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    samples = load_dataset(args)
    if not samples:
        raise SystemExit("평가할 샘플이 없습니다.")

    if args.baseline:
        run_baseline(args.baseline, samples, args.output)
        return

    generate_explanation = not args.no_explanation
    run(
        samples=samples,
        output_path=args.output,
        variant=args.variant,
        score_explanation=generate_explanation and not args.no_judge,
        generate_explanation=generate_explanation,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
