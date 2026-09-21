"""판정과 분리된 Toulmin 설명 검증 (논문 3.5절)."""

import pytest

from prism.agents.generator_toulmin import (
    TOULMIN_ELEMENTS,
    render_report,
    structural_completeness,
)

FULL = {key: f"{key} 내용" for key in TOULMIN_ELEMENTS}


def test_six_elements_are_fixed():
    assert TOULMIN_ELEMENTS == (
        "claim", "grounds", "warrant", "backing", "qualifier", "rebuttal"
    )


class TestStructuralCompleteness:
    def test_all_elements_filled(self):
        assert structural_completeness(FULL) == 1.0

    def test_missing_elements_lower_the_score(self):
        partial = dict(FULL)
        del partial["rebuttal"]
        del partial["backing"]
        assert structural_completeness(partial) == pytest.approx(4 / 6)

    def test_whitespace_only_counts_as_missing(self):
        assert structural_completeness({**FULL, "warrant": "   "}) == pytest.approx(5 / 6)

    def test_empty_explanation(self):
        assert structural_completeness({}) == 0.0


class TestRenderReport:
    def test_leads_with_the_verdict(self):
        assert render_report("Refuted", FULL).startswith("판정: Refuted")

    def test_contains_every_element(self):
        report = render_report("Supported", FULL)
        for key in TOULMIN_ELEMENTS:
            assert f"{key} 내용" in report

    def test_marks_missing_elements_instead_of_dropping_them(self):
        # 요소가 비어도 자리를 남겨야 구조 완전성을 눈으로 확인할 수 있다.
        report = render_report("NEI", {**FULL, "rebuttal": ""})
        assert "(누락)" in report

    def test_does_not_invent_a_verdict(self):
        # 렌더링은 결정적이며 LLM을 다시 부르지 않는다. 주어진 판정만 쓴다.
        assert "Supported" not in render_report("Refuted", FULL)
