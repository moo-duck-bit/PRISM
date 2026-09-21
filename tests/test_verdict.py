from collections import Counter

import pytest

from prism.core.verdict import (
    NEI,
    REFUTED,
    SUPPORTED,
    TEXT_VERIFIERS,
    majority_label,
    normalize_verdict,
    plurality_label,
    quorum_label,
    vote_counts,
)


def v(label, confidence=0.9):
    return {"verdict": label, "confidence": confidence, "rationale": "-"}


class TestNormalizeVerdict:
    @pytest.mark.parametrize("raw", ["Supported", "supported", "SUPPORTED", "true", "support"])
    def test_accepts_known_spellings(self, raw):
        assert normalize_verdict({"verdict": raw})["verdict"] == SUPPORTED

    @pytest.mark.parametrize("raw", ["unproven", "abstain", "not enough info"])
    def test_maps_benchmark_abstentions_to_nei(self, raw):
        assert normalize_verdict({"verdict": raw})["verdict"] == NEI

    def test_unknown_label_falls_back_to_nei(self):
        assert normalize_verdict({"verdict": "Mostly True"})["verdict"] == NEI

    def test_non_numeric_confidence_becomes_zero(self):
        assert normalize_verdict({"verdict": "Refuted", "confidence": "high"})["confidence"] == 0.0

    @pytest.mark.parametrize("value,expected", [(1.7, 1.0), (-0.4, 0.0), (0.42, 0.42)])
    def test_confidence_is_clamped(self, value, expected):
        # gamma_v 는 임계값과 직접 비교되므로 범위를 벗어난 값이 새어 나가면 안 된다.
        assert normalize_verdict({"verdict": "Refuted", "confidence": value})["confidence"] == expected

    def test_missing_rationale_becomes_empty_string(self):
        assert normalize_verdict({"verdict": "NEI", "rationale": None})["rationale"] == ""


class TestVoteCounts:
    def test_counts_all_verifiers_by_default(self):
        verdicts = {"factual": v(SUPPORTED), "entity": v(SUPPORTED), "visual": v(REFUTED)}
        assert vote_counts(verdicts) == Counter({SUPPORTED: 2, REFUTED: 1})

    def test_can_restrict_to_text_verifiers(self):
        verdicts = {"factual": v(SUPPORTED), "entity": v(SUPPORTED), "visual": v(REFUTED)}
        assert vote_counts(verdicts, TEXT_VERIFIERS) == Counter({SUPPORTED: 2})

    def test_ignores_non_dict_entries(self):
        assert vote_counts({"factual": v(REFUTED), "broken": "oops"}) == Counter({REFUTED: 1})


class TestQuorum:
    def test_returns_label_reaching_the_quorum(self):
        assert quorum_label(Counter({SUPPORTED: 3, REFUTED: 1}), 3) == SUPPORTED

    def test_returns_none_below_the_quorum(self):
        assert quorum_label(Counter({SUPPORTED: 2, REFUTED: 1}), 3) is None


class TestMajority:
    def test_strict_majority(self):
        assert majority_label(Counter({REFUTED: 2, SUPPORTED: 1})) == REFUTED

    def test_exact_half_is_not_a_majority(self):
        assert majority_label(Counter({REFUTED: 2, SUPPORTED: 2})) is None

    def test_empty(self):
        assert majority_label(Counter()) is None


class TestPlurality:
    def test_top_label_wins(self):
        assert plurality_label(Counter({NEI: 2, SUPPORTED: 1})) == NEI

    def test_tie_uses_the_tie_breaker(self):
        assert plurality_label(Counter({SUPPORTED: 1, REFUTED: 1})) == NEI
        assert plurality_label(Counter({SUPPORTED: 1, REFUTED: 1}), SUPPORTED) == SUPPORTED

    def test_empty_counts(self):
        assert plurality_label(Counter()) == NEI
