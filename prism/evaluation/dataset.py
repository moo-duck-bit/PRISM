"""벤치마크 로더.

두 벤치마크를 같은 Sample 형태로 맞춘다.

MOCHEG-Politics
    PolitiFact 서브셋. 두 설정으로 평가한다.
    - binary: Supported/Refuted 균형 세트, NEI 제외 (선행 연구와 같은 프로토콜)
    - three_class: NEI를 포함한 3-class 전체
    D_c 는 claim_id 로 연결된 Corpus3 문서이며, 검색 에이전트가 직접 읽는다.

RW-Post Politics
    실제 소셜 미디어 정치 클레임. 벤치마크가 gold fact-check 증거를 제공하므로
    그 스니펫을 D_c 로 넘긴다.
    정답 누출을 막기 위해 fact-checker의 결론 필드(reasoning_logic, key_points)는
    읽지 않는다. LEAKING_FIELDS 에 명시해 두고 로더가 강제한다.

샘플 순서는 섞지 않는다. resume이 순서에 기대기 때문이다.
층화 추출이 필요한 경우에만 시드를 고정해 정렬한다.
"""

import csv
import json
import logging
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from prism.evaluation.metrics import canonical

logger = logging.getLogger(__name__)

# RW-Post에서 절대 읽지 않는 필드. fact-checker의 결론이 들어 있다.
LEAKING_FIELDS = ("reasoning_logic", "key_points", "verdict", "label_explanation")


@dataclass
class Sample:
    claim_id: str
    claim: str
    label: str                                   # 정규화된 Supported/Refuted/NEI
    post_text: str = ""
    ocr_text: str = ""
    image_paths: List[str] = field(default_factory=list)
    evidence_pool: List[str] = field(default_factory=list)


def _read_claim_texts(claims_csv: str) -> Dict[str, str]:
    if not os.path.exists(claims_csv):
        raise FileNotFoundError(f"클레임 원문 CSV를 찾을 수 없습니다: {claims_csv}")

    texts: Dict[str, str] = {}
    with open(claims_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.setdefault(row["claim_id"].strip(), row["Claim"].strip())
    return texts


def _stratify(samples: List[Sample], n_per_class: int, seed: int) -> List[Sample]:
    """클래스별로 n개씩 뽑는다. 시드를 고정해 같은 목록이 재현되게 한다."""
    by_label: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        by_label[sample.label].append(sample)

    rng = random.Random(seed)
    selected: List[Sample] = []
    for label in sorted(by_label):
        pool = by_label[label]
        if len(pool) <= n_per_class:
            logger.warning("%s 클래스가 %d개뿐입니다 (요청 %d)", label, len(pool), n_per_class)
            selected.extend(pool)
        else:
            selected.extend(rng.sample(pool, n_per_class))

    # 원래 순서를 복원해 resume 일관성을 지킨다.
    order = {id(s): i for i, s in enumerate(samples)}
    return sorted(selected, key=lambda s: order[id(s)])


def load_mocheg(
    labels_csv: str,
    claims_csv: str,
    setting: str = "three_class",
    n_per_class: Optional[int] = None,
    seed: int = 42,
) -> List[Sample]:
    """MOCHEG-Politics 샘플을 만든다.

    Args:
        labels_csv: claim_id, label 컬럼을 가진 CSV.
        claims_csv: claim_id, Claim 컬럼을 가진 CSV (Corpus2).
        setting: "binary"면 NEI를 제외하고, "three_class"면 전부 쓴다.
        n_per_class: 클래스별 표본 수. binary 150이면 균형 300 세트가 된다.
    """
    if setting not in ("binary", "three_class"):
        raise ValueError(f"알 수 없는 setting: {setting!r}")

    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"라벨 CSV를 찾을 수 없습니다: {labels_csv}")

    texts = _read_claim_texts(claims_csv)
    samples: List[Sample] = []
    missing = 0

    with open(labels_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            claim_id = row["claim_id"].strip()
            claim = texts.get(claim_id, "")
            if not claim:
                missing += 1
                continue

            label = canonical(row["label"])
            if setting == "binary" and label == "NEI":
                continue

            samples.append(Sample(claim_id=claim_id, claim=claim, label=label))

    if missing:
        logger.warning("클레임 원문을 찾지 못한 샘플 %d건을 제외했습니다", missing)

    if n_per_class is not None:
        samples = _stratify(samples, n_per_class, seed)

    logger.info("MOCHEG %s: %d건 %s", setting, len(samples),
                dict(Counter(s.label for s in samples)))
    return samples


def _evidence_snippets(record: Dict[str, Any], field_name: str) -> List[str]:
    """gold 증거를 문자열 스니펫 리스트로 펼친다."""
    raw = record.get(field_name) or []
    if isinstance(raw, str):
        return [raw]

    snippets: List[str] = []
    for item in raw:
        if isinstance(item, str):
            snippets.append(item)
        elif isinstance(item, dict):
            # 항목이 dict면 본문에 해당하는 값만 모은다.
            for key in ("content", "text", "snippet", "evidence", "body"):
                if item.get(key):
                    snippets.append(str(item[key]))
                    break
    return snippets


def load_rwpost(
    path: str,
    image_root: str = "",
    evidence_field: str = "fact_checking_evidence",
    closed_book: bool = False,
) -> List[Sample]:
    """RW-Post Politics 샘플을 만든다.

    Args:
        path: JSON 배열 또는 JSONL 파일.
        image_root: 이미지 경로의 기준 디렉터리.
        closed_book: True면 증거 풀을 비운다(게이팅 제거 설정).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"RW-Post 파일을 찾을 수 없습니다: {path}")

    with open(path, encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        records = json.load(f) if first == "[" else [json.loads(line) for line in f if line.strip()]

    samples: List[Sample] = []
    for i, record in enumerate(records):
        claim = str(record.get("claim") or record.get("text") or "").strip()
        if not claim:
            continue

        images = record.get("image_paths") or record.get("images") or []
        if isinstance(images, str):
            images = [images]
        if image_root:
            images = [os.path.join(image_root, os.path.basename(p)) for p in images]

        samples.append(Sample(
            claim_id=str(record.get("id", record.get("claim_id", i))),
            claim=claim,
            label=canonical(record.get("label", record.get("gold_label", ""))),
            post_text=str(record.get("post_text") or record.get("post") or ""),
            ocr_text=str(record.get("ocr") or record.get("ocr_text") or ""),
            image_paths=list(images),
            evidence_pool=[] if closed_book else _evidence_snippets(record, evidence_field),
        ))

    leaked = LEAKING_FIELDS if records and any(k in records[0] for k in LEAKING_FIELDS) else ()
    if leaked:
        logger.info("정답 누출 방지: %s 필드는 읽지 않습니다", ", ".join(leaked))

    logger.info("RW-Post: %d건 %s (closed_book=%s)", len(samples),
                dict(Counter(s.label for s in samples)), closed_book)
    return samples
