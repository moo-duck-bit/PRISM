import json

import pytest

from prism.evaluation.dataset import LEAKING_FIELDS, load_mocheg, load_rwpost

LABELS_CSV = """claim_id,label
1,supported
2,refuted
3,nei
4,refuted
5,supported
6,supported
"""

CLAIMS_CSV = """claim_id,Claim
1,First claim
2,Second claim
3,Third claim
4,Fourth claim
5,Fifth claim
"""


@pytest.fixture
def mocheg_files(tmp_path):
    labels = tmp_path / "labels.csv"
    claims = tmp_path / "claims.csv"
    labels.write_text(LABELS_CSV, encoding="utf-8")
    claims.write_text(CLAIMS_CSV, encoding="utf-8")
    return str(labels), str(claims)


class TestMocheg:
    def test_joins_labels_with_claim_text(self, mocheg_files):
        sample = load_mocheg(*mocheg_files)[0]
        assert (sample.claim_id, sample.claim, sample.label) == ("1", "First claim", "Supported")

    def test_binary_setting_drops_nei(self, mocheg_files):
        labels = [s.label for s in load_mocheg(*mocheg_files, setting="binary")]
        assert "NEI" not in labels

    def test_three_class_keeps_nei(self, mocheg_files):
        labels = [s.label for s in load_mocheg(*mocheg_files, setting="three_class")]
        assert "NEI" in labels

    def test_skips_claims_without_text(self, mocheg_files):
        # claim_id 6은 원문 CSV에 없다.
        assert all(s.claim_id != "6" for s in load_mocheg(*mocheg_files))

    def test_preserves_file_order(self, mocheg_files):
        # resume이 순서에 기대므로 셔플이 끼어들면 안 된다.
        assert [s.claim_id for s in load_mocheg(*mocheg_files)] == ["1", "2", "3", "4", "5"]

    def test_stratified_sampling_balances_classes(self, mocheg_files):
        samples = load_mocheg(*mocheg_files, setting="binary", n_per_class=2)
        labels = [s.label for s in samples]
        assert labels.count("Supported") == 2 and labels.count("Refuted") == 2

    def test_stratified_sampling_is_reproducible(self, mocheg_files):
        first = load_mocheg(*mocheg_files, setting="binary", n_per_class=1, seed=7)
        second = load_mocheg(*mocheg_files, setting="binary", n_per_class=1, seed=7)
        assert [s.claim_id for s in first] == [s.claim_id for s in second]

    def test_stratified_sampling_keeps_file_order(self, mocheg_files):
        ids = [s.claim_id for s in load_mocheg(*mocheg_files, setting="binary", n_per_class=2)]
        assert ids == sorted(ids, key=int)

    def test_rejects_unknown_setting(self, mocheg_files):
        with pytest.raises(ValueError):
            load_mocheg(*mocheg_files, setting="four_class")

    def test_missing_file_raises(self, tmp_path, mocheg_files):
        _labels, claims = mocheg_files
        with pytest.raises(FileNotFoundError):
            load_mocheg(str(tmp_path / "nope.csv"), claims)


RWPOST = [
    {
        "id": "p1",
        "claim": "The senator said the bill was dead.",
        "label": "FALSE",
        "post_text": "BREAKING",
        "ocr": "DEAD ON ARRIVAL",
        "images": ["p1.jpg"],
        "fact_checking_evidence": [
            {"content": "The senator voted for the bill."},
            {"content": "The bill passed the chamber."},
        ],
        "reasoning_logic": "LEAK: the fact-checker concluded this is false",
        "key_points": "LEAK",
    },
    {
        "id": "p2",
        "claim": "A rally drew a record crowd.",
        "label": "UNPROVEN",
        "fact_checking_evidence": ["Attendance figures were not released."],
    },
]


@pytest.fixture
def rwpost_file(tmp_path):
    path = tmp_path / "rwpost.json"
    path.write_text(json.dumps(RWPOST), encoding="utf-8")
    return str(path)


class TestRwPost:
    def test_normalizes_benchmark_labels(self, rwpost_file):
        labels = [s.label for s in load_rwpost(rwpost_file)]
        assert labels == ["Refuted", "NEI"]

    def test_carries_post_text_and_ocr(self, rwpost_file):
        sample = load_rwpost(rwpost_file)[0]
        assert sample.post_text == "BREAKING"
        assert sample.ocr_text == "DEAD ON ARRIVAL"

    def test_flattens_dict_evidence_into_snippets(self, rwpost_file):
        assert load_rwpost(rwpost_file)[0].evidence_pool == [
            "The senator voted for the bill.",
            "The bill passed the chamber.",
        ]

    def test_accepts_plain_string_evidence(self, rwpost_file):
        assert load_rwpost(rwpost_file)[1].evidence_pool == [
            "Attendance figures were not released."
        ]

    def test_never_reads_the_fact_checker_conclusion(self, rwpost_file):
        # 정답 누출 방지. 결론 필드가 증거 풀이나 본문으로 새면 안 된다.
        sample = load_rwpost(rwpost_file)[0]
        blob = " ".join(sample.evidence_pool + [sample.claim, sample.post_text, sample.ocr_text])
        assert "LEAK" not in blob
        assert "reasoning_logic" in LEAKING_FIELDS and "key_points" in LEAKING_FIELDS

    def test_closed_book_empties_the_evidence_pool(self, rwpost_file):
        assert all(s.evidence_pool == [] for s in load_rwpost(rwpost_file, closed_book=True))

    def test_image_root_is_applied(self, rwpost_file):
        sample = load_rwpost(rwpost_file, image_root="/images")[0]
        assert sample.image_paths[0].replace("\\", "/") == "/images/p1.jpg"

    def test_reads_jsonl(self, tmp_path):
        path = tmp_path / "rwpost.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in RWPOST), encoding="utf-8")
        assert len(load_rwpost(str(path))) == 2

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_rwpost(str(tmp_path / "nope.json"))
