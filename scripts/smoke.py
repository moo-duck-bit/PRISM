"""에이전트를 하나씩, 또는 파이프라인 전체를 실제 API로 한 번 돌려 보는 점검 스크립트.

tests/ 아래 단위 테스트는 네트워크를 타지 않는다. 프롬프트를 고친 뒤
모델이 실제로 어떤 응답을 주는지 눈으로 확인할 때 이 스크립트를 쓴다.
OPENAI_API_KEY가 필요하고, 호출한 만큼 비용이 발생한다.

사용 예:
    python scripts/smoke.py triage
    python scripts/smoke.py factual entity spatiotemporal
    python scripts/smoke.py visual --image data/post.jpg
    python scripts/smoke.py pipeline --variant no_calibrator
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism.agents.generator_toulmin import toulmin_generator_agent  # noqa: E402
from prism.agents.retrieval_text import textual_retrieval_agent  # noqa: E402
from prism.agents.retrieval_visual import visual_retrieval_agent  # noqa: E402
from prism.agents.triage import triage_agent  # noqa: E402
from prism.agents.verifier_entity import entity_verifier  # noqa: E402
from prism.agents.verifier_factual import factual_verifier  # noqa: E402
from prism.agents.verifier_spatiotemporal import spatiotemporal_verifier  # noqa: E402
from prism.agents.verifier_visual import visual_verifier  # noqa: E402
from prism.core.calibrator import calibrate  # noqa: E402
from prism.core.graph import ABLATIONS, build_variant, initial_state  # noqa: E402

# 인물, 장소, 시각이 모두 들어 있어 검증기 네 종류를 한 번에 자극하는 예시다.
CLAIM = (
    "A man was beaten to death outside Lauren Boebert's restaurant "
    "in Rifle, Colorado on a Friday night."
)

# 사실관계는 반박되고 장소는 일치하는 증거. 검증기별로 다른 판정이 나와야 정상이다.
EVIDENCE = [
    "[E1] Official autopsy reports show Anthony Royal Green died from a methamphetamine overdose.",
    "[E2] The incident occurred near Shooters Grill in Rifle, Colorado.",
    "[E3] Police records show the event happened on a Wednesday morning, not Friday night.",
]


def _state(args: argparse.Namespace, with_evidence: bool = True) -> Dict:
    state = initial_state(
        claim=args.claim,
        claim_id=args.claim_id,
        ocr_text=args.ocr,
        image_paths=args.image,
    )
    if with_evidence:
        state["text_evidence"] = list(EVIDENCE)
    # 검증기를 단독으로 부를 때는 triage 결과가 없으므로 마스크를 열어 둔다.
    state["modality_mask"] = {"img": bool(args.image), "ent": True, "time": True}
    state["image_evidence"] = [p for p in args.image if os.path.exists(p)]
    return state


def _dump(title: str, payload) -> None:
    print(f"\n--- {title} ---")
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def run_triage(args) -> None:
    result = triage_agent(_state(args))
    _dump("mask", result["modality_mask"])
    _dump("S(m)", result["active_agents"])


def run_retrieval(args) -> None:
    """검색만 따로 돌린다. 코퍼스 CSV가 필요하고 LLM은 호출하지 않는다."""
    result = textual_retrieval_agent(_state(args, with_evidence=False))
    for evidence in result["text_evidence"]:
        print(f"\n{evidence[:300]}")


def run_visual_retrieval(args) -> None:
    _dump("E_I", visual_retrieval_agent(_state(args))["image_evidence"])


def run_factual(args) -> None:
    _dump("f", factual_verifier(_state(args))["verdicts"]["factual"])


def run_entity(args) -> None:
    _dump("e", entity_verifier(_state(args))["verdicts"]["entity"])


def run_spatiotemporal(args) -> None:
    _dump("st", spatiotemporal_verifier(_state(args))["verdicts"]["spatiotemporal"])


def run_visual(args) -> None:
    if not args.image:
        print("--image 로 경로를 지정하지 않아 '이미지 없음' 경로를 확인합니다.")
    _dump("v", visual_verifier(_state(args))["verdicts"]["visual"])


def run_calibrator(args) -> None:
    """교차모달 충돌을 인위적으로 만들어 delta_theta 가 어떻게 도는지 본다."""
    state = _state(args)
    state["verdicts"] = {
        "factual": {"verdict": "Supported", "confidence": 0.8, "rationale": "text agrees"},
        "entity": {"verdict": "Supported", "confidence": 0.7, "rationale": "entity matches"},
        "visual": {"verdict": "Refuted", "confidence": 0.9, "rationale": "image is from another event"},
    }
    state["image_evidence"] = ["synthetic.jpg"]
    _dump("calibration", calibrate(state))


def run_explanation(args) -> None:
    state = _state(args)
    state["verdict"] = "Refuted"
    state["verdicts"] = {
        "factual": {"verdict": "Refuted", "confidence": 0.95,
                    "rationale": "The autopsy states an overdose, not a beating."},
    }
    result = toulmin_generator_agent(state)
    _dump("toulmin", result["explanation"])
    print(f"\n{result['report']}")


def run_pipeline(args) -> None:
    app = build_variant(args.variant)
    final = app.invoke(_state(args, with_evidence=False))
    _dump("mask", final.get("modality_mask", {}))
    _dump("verdicts", final.get("verdicts", {}))
    _dump("calibration", final.get("calibration", {}))
    print(f"\n최종 판정: {final.get('verdict')}")
    if final.get("report"):
        print(f"\n{final['report']}")


TARGETS = {
    "triage": run_triage,
    "retrieval": run_retrieval,
    "visual_retrieval": run_visual_retrieval,
    "factual": run_factual,
    "entity": run_entity,
    "spatiotemporal": run_spatiotemporal,
    "visual": run_visual,
    "calibrator": run_calibrator,
    "explanation": run_explanation,
    "pipeline": run_pipeline,
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PRISM 에이전트 수동 점검")
    parser.add_argument("targets", nargs="+", choices=sorted(TARGETS))
    parser.add_argument("--claim", default=CLAIM)
    parser.add_argument("--claim-id", default="")
    parser.add_argument("--ocr", default="")
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--variant", choices=tuple(ABLATIONS), default="full")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    print(f"Claim: {args.claim}")
    for target in args.targets:
        print(f"\n=== {target} ===")
        TARGETS[target](args)


if __name__ == "__main__":
    main()
