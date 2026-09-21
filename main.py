"""단일 클레임을 PRISM 파이프라인에 태우는 실행 진입점.

사용 예:
    python main.py --claim "..."
    python main.py --claim "..." --image data/post.jpg --ocr "BREAKING"
    python main.py --claim "..." --variant no_gating --trace
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Optional

# 저장소 루트를 경로에 넣어 어디서 실행하든 prism 패키지를 찾게 한다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prism.core.graph import ABLATIONS, build_variant, initial_state  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PRISM 멀티모달 팩트체킹 파이프라인")
    parser.add_argument("--claim", required=True, help="검증할 문장 (c)")
    parser.add_argument("--claim-id", default="", help="증거 풀 D_c 를 좁히는 식별자")
    parser.add_argument("--post", default="", help="포스트 본문 (P)")
    parser.add_argument("--ocr", default="", help="이미지에서 추출한 텍스트 (O)")
    parser.add_argument("--image", action="append", default=[], help="이미지 경로 (I). 반복 지정 가능")
    parser.add_argument("--variant", choices=tuple(ABLATIONS), default="full")
    parser.add_argument("--no-explanation", action="store_true",
                        help="Toulmin 생성을 건너뛰고 판정까지만 수행")
    parser.add_argument("--trace", action="store_true", help="캘리브레이션 경로를 출력")
    parser.add_argument("--quiet", action="store_true", help="진행 로그를 숨긴다")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    app = build_variant(args.variant, generate_explanation=not args.no_explanation)
    state = app.invoke(initial_state(
        claim=args.claim,
        claim_id=args.claim_id,
        post_text=args.post,
        ocr_text=args.ocr,
        image_paths=args.image,
    ))

    mask = state.get("modality_mask", {})
    active = state.get("active_agents", [])

    print(f"\nClaim: {args.claim}")
    print(f"Mask  img={int(bool(mask.get('img')))} "
          f"ent={int(bool(mask.get('ent')))} "
          f"time={int(bool(mask.get('time')))}  ->  활성 {len(active)}/4 {active}")
    print(f"증거  E_T={len(state.get('text_evidence', []))}건  "
          f"E_I={len(state.get('image_evidence', []))}건\n")

    for name, result in state.get("verdicts", {}).items():
        print(f"  {name:<16} {result.get('verdict'):<10} "
              f"conf={result.get('confidence'):.2f}  {result.get('rationale')}")

    print(f"\n최종 판정: {state.get('verdict')}")

    if args.trace:
        print(f"\n캘리브레이션: {json.dumps(state.get('calibration', {}), ensure_ascii=False)}")

    if state.get("report"):
        print(f"\n{state['report']}")


if __name__ == "__main__":
    main()
