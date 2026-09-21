import pytest

from prism.evaluation.metrics import (
    accuracy,
    bootstrap_ci,
    canonical,
    evaluate,
    macro_f1,
    mcnemar,
    nei_recall,
    refuted_to_supported,
)

S, R, N = "Supported", "Refuted", "NEI"


class TestCanonical:
    @pytest.mark.parametrize("raw,expected", [
        ("supported", S), ("TRUE", S),
        ("refuted", R), ("False", R),
        ("nei", N), ("UNPROVEN", N),
    ])
    def test_benchmark_labels_map_to_internal_labels(self, raw, expected):
        # RW-Post는 TRUE/FALSE/UNPROVEN, MOCHEG는 supported/refuted/nei로 표기한다.
        assert canonical(raw) == expected

    def test_unknown_label_becomes_nei(self):
        assert canonical("half-true") == N


class TestAccuracyAndF1:
    def test_accuracy(self):
        assert accuracy([S, R, N], [S, R, S]) == pytest.approx(2 / 3)

    def test_empty_input(self):
        assert accuracy([], []) == 0.0

    def test_macro_f1_is_perfect_on_perfect_predictions(self):
        assert macro_f1([S, R, N], [S, R, N]) == 1.0

    def test_macro_f1_ignores_classes_absent_from_the_gold_labels(self):
        # 이진 세트에서는 NEI가 정답에 없으므로 macro 평균에 들어가지 않는다.
        assert macro_f1([S, S, R, R], [S, S, R, R]) == 1.0

    def test_macro_f1_penalizes_a_collapsed_predictor(self):
        # 전부 Supported로 찍으면 F1(S)=0.667, F1(R)=0 이므로 macro는 0.333이다.
        assert macro_f1([S, S, R, R], [S, S, S, S]) == pytest.approx(1 / 3)


class TestSafetyMetrics:
    def test_r_to_s_counts_only_refuted_gold(self):
        # 거짓인 주장 4건 중 1건을 사실로 판정했다.
        assert refuted_to_supported([R, R, R, R, S], [S, R, R, N, S]) == 0.25

    def test_r_to_s_is_zero_without_refuted_gold(self):
        assert refuted_to_supported([S, S], [S, R]) == 0.0

    def test_nei_recall(self):
        assert nei_recall([N, N, S], [N, S, S]) == 0.5

    def test_nei_recall_is_zero_when_the_system_never_abstains(self):
        # 캘리브레이터를 제거했을 때 나타나는 상태다.
        assert nei_recall([N, N], [S, R]) == 0.0


class TestEvaluate:
    def test_reports_all_metrics(self):
        scores = evaluate(["supported", "refuted", "nei"], ["Supported", "Supported", "NEI"])
        assert scores["n"] == 3
        assert scores["accuracy"] == pytest.approx(2 / 3)
        assert scores["r_to_s"] == 1.0
        assert scores["nei_recall"] == 1.0

    def test_normalizes_mixed_label_spellings(self):
        scores = evaluate(["TRUE", "FALSE"], ["supported", "refuted"])
        assert scores["accuracy"] == 1.0


class TestStatistics:
    def test_bootstrap_ci_brackets_the_point_estimate(self):
        y_true = [S, R] * 25
        y_pred = [S, S] * 25
        lower, upper = bootstrap_ci(y_true, y_pred, "accuracy", iterations=200)
        assert lower <= 0.5 <= upper

    def test_bootstrap_ci_on_empty_input(self):
        assert bootstrap_ci([], [], "accuracy") == [0.0, 0.0]

    def test_mcnemar_is_insignificant_for_identical_systems(self):
        y_true = [S, R, N, S]
        assert mcnemar(y_true, y_true, y_true)["p_value"] == 1.0

    def test_mcnemar_counts_discordant_pairs(self):
        y_true = [S] * 10
        a = [S] * 10
        b = [R] * 10
        result = mcnemar(y_true, a, b)
        assert result["b01"] == 10 and result["b10"] == 0
        assert result["p_value"] < 0.01

    def test_mcnemar_is_insignificant_on_few_discordant_pairs(self):
        # 연속성 보정 때문에 표본이 작으면 같은 방향이어도 유의하지 않다.
        assert mcnemar([S] * 4, [S] * 4, [R] * 4)["p_value"] > 0.05
