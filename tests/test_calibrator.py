"""Deterministic-First Cross-Modal Calibration 검증 (논문 3.4절).

LLM 위임 경로는 monkeypatch로 끊고, 규칙이 실제로 무엇을 결정하는지만 본다.
"""

import pytest

from prism.core import calibrator
from prism.core.calibrator import (
    BOTTOM,
    Theta,
    calibrate,
    conflict,
    delta_theta,
    psi_guard,
)
from prism.core.verdict import NEI, REFUTED, SUPPORTED

THETA = Theta(tau_v=0.7, tau_q=3)


def v(label, confidence=0.8, rationale="-"):
    return {"verdict": label, "confidence": confidence, "rationale": rationale}


def state(verdicts, with_image=True):
    return {
        "claim": "test claim",
        "verdicts": verdicts,
        "image_evidence": ["img.jpg"] if with_image else [],
    }


@pytest.fixture
def no_llm(monkeypatch):
    """LLM 위임이 일어나면 테스트가 알아채도록 표시를 남긴다."""
    calls = []

    def fake(_state):
        calls.append(_state)
        return NEI, "llm called"

    monkeypatch.setattr(calibrator, "_llm_verdict", fake)
    return calls


class TestConflict:
    def test_paper_pattern_is_a_conflict(self):
        # 텍스트 2표 Supported + 반박 0 + 시각 Refuted + 이미지 존재
        s = state({"factual": v(SUPPORTED), "entity": v(SUPPORTED), "visual": v(REFUTED, 0.9)})
        assert conflict(s) is True

    def test_needs_at_least_two_supporting_text_verifiers(self):
        s = state({"factual": v(SUPPORTED), "visual": v(REFUTED, 0.9)})
        assert conflict(s) is False

    def test_any_text_refutation_cancels_the_conflict(self):
        s = state({
            "factual": v(SUPPORTED), "entity": v(SUPPORTED),
            "spatiotemporal": v(REFUTED), "visual": v(REFUTED, 0.9),
        })
        assert conflict(s) is False

    def test_requires_a_refuting_visual_verdict(self):
        s = state({"factual": v(SUPPORTED), "entity": v(SUPPORTED), "visual": v(SUPPORTED)})
        assert conflict(s) is False

    def test_requires_image_evidence(self):
        s = state(
            {"factual": v(SUPPORTED), "entity": v(SUPPORTED), "visual": v(REFUTED, 0.9)},
            with_image=False,
        )
        assert conflict(s) is False


class TestDeltaTheta:
    def test_high_confidence_visual_overrides_text_consensus(self):
        # 논문 Figure 1의 사례. 텍스트는 Supported, 시각이 0.90으로 반박한다.
        s = state({"factual": v(SUPPORTED), "entity": v(SUPPORTED), "visual": v(REFUTED, 0.90)})
        assert delta_theta(s, THETA) == REFUTED

    def test_low_confidence_visual_abstains_instead(self):
        s = state({"factual": v(SUPPORTED), "entity": v(SUPPORTED), "visual": v(REFUTED, 0.55)})
        assert delta_theta(s, THETA) == NEI

    def test_threshold_is_inclusive(self):
        s = state({"factual": v(SUPPORTED), "entity": v(SUPPORTED), "visual": v(REFUTED, 0.70)})
        assert delta_theta(s, THETA) == REFUTED

    def test_quorum_decides_when_there_is_no_conflict(self):
        s = state({
            "factual": v(REFUTED), "entity": v(REFUTED),
            "spatiotemporal": v(REFUTED), "visual": v(NEI, 0.0),
        })
        assert delta_theta(s, THETA) == REFUTED

    def test_modality_conflict_abstains(self):
        # 텍스트 과반은 Refuted, 시각은 Supported. conflict() 패턴은 아니다.
        s = state({"factual": v(REFUTED), "entity": v(REFUTED), "visual": v(SUPPORTED, 0.8)})
        assert delta_theta(s, THETA) == NEI

    def test_returns_bottom_when_no_rule_applies(self):
        s = state({"factual": v(SUPPORTED), "entity": v(NEI, 0.0)}, with_image=False)
        assert delta_theta(s, THETA) is BOTTOM


class TestPsiGuard:
    def test_blocks_refuted_to_supported_promotion(self):
        s = state({"factual": v(SUPPORTED), "entity": v(SUPPORTED), "visual": v(REFUTED, 0.9)})
        verdict, rule = psi_guard(SUPPORTED, s, THETA)
        assert verdict == REFUTED
        assert rule == "guard_block_rs"

    def test_blocked_promotion_abstains_when_visual_is_weak(self):
        s = state({"factual": v(SUPPORTED), "entity": v(SUPPORTED), "visual": v(REFUTED, 0.5)})
        verdict, _rule = psi_guard(SUPPORTED, s, THETA)
        assert verdict == NEI

    def test_quorum_overrides_the_llm(self):
        s = state({
            "factual": v(NEI, 0.0), "entity": v(NEI, 0.0), "spatiotemporal": v(NEI, 0.0),
        }, with_image=False)
        verdict, rule = psi_guard(SUPPORTED, s, THETA)
        assert verdict == NEI
        assert rule == "guard_quorum"

    def test_majority_overrides_the_llm(self):
        s = state({"factual": v(REFUTED), "entity": v(REFUTED), "spatiotemporal": v(SUPPORTED)},
                  with_image=False)
        verdict, rule = psi_guard(SUPPORTED, s, THETA)
        assert verdict == REFUTED
        assert rule == "guard_majority"

    def test_llm_stands_when_verifiers_are_split(self):
        s = state({"factual": v(SUPPORTED), "entity": v(REFUTED)}, with_image=False)
        verdict, rule = psi_guard(NEI, s, THETA)
        assert verdict == NEI
        assert rule == "llm"


class TestCalibrate:
    def test_multimodal_conflict_is_resolved_without_the_llm(self, no_llm):
        s = state({"factual": v(SUPPORTED), "entity": v(SUPPORTED), "visual": v(REFUTED, 0.9)})
        result = calibrate(s)
        assert result["verdict"] == REFUTED
        assert result["calibration"]["deterministic"] is True
        assert result["calibration"]["path"] == "multimodal"
        assert no_llm == []

    def test_text_dominant_path_goes_through_the_llm_and_the_guard(self, no_llm):
        s = state({"factual": v(SUPPORTED), "entity": v(REFUTED)}, with_image=False)
        result = calibrate(s)
        assert result["calibration"]["path"] == "text_dominant"
        assert len(no_llm) == 1

    def test_disabling_rules_hands_the_verdict_to_the_llm(self, no_llm):
        # 캘리브레이터 제거 ablation. 규칙이 잡아 주던 충돌이 그대로 통과한다.
        s = state({"factual": v(SUPPORTED), "entity": v(SUPPORTED), "visual": v(REFUTED, 0.9)})
        result = calibrate(s, use_rules=False)
        assert result["calibration"]["rule"] == "disabled"
        assert len(no_llm) == 1

    def test_trace_records_the_thresholds(self, no_llm):
        result = calibrate(state({"factual": v(SUPPORTED)}, with_image=False))
        assert result["calibration"]["theta"] == {"tau_v": 0.7, "tau_q": 3}
