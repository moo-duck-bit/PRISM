"""그래프 수준 compute gating 검증 (논문 3.2절).

비활성 에이전트는 프롬프트 안에서 건너뛰는 것이 아니라 노드 자체가 실행되지 않는다.
라우터가 그 건너뛰기를 담당하므로, 라우터 동작을 직접 확인한다.
"""

import pytest

from prism.core.graph import (
    ABLATIONS,
    CALIBRATE,
    VERIFIER_NODES,
    _bypass_triage,
    _route_targets,
    _router,
    initial_state,
)

NODE_NAMES = [node for node, _agent, _fn in VERIFIER_NODES]


def route(after_index, active):
    return _router(after_index, CALIBRATE)({"active_agents": active})


class TestRouter:
    def test_skips_inactive_agents(self):
        # f 다음에 e와 st가 꺼져 있으면 바로 v로 건너뛴다.
        assert route(0, ["factual", "visual"]) == "verify_visual"

    def test_goes_to_the_next_active_agent(self):
        assert route(0, ["factual", "entity", "visual"]) == "verify_entity"

    def test_terminates_when_nothing_is_left(self):
        assert route(0, ["factual"]) == CALIBRATE

    def test_terminates_from_the_middle_of_the_chain(self):
        assert route(1, ["factual", "entity"]) == CALIBRATE

    def test_never_routes_backwards(self):
        # index 2 이후에서 앞쪽 노드로 돌아가면 무한 루프가 된다.
        assert route(2, ["factual", "entity", "spatiotemporal", "visual"]) == "verify_visual"

    @pytest.mark.parametrize("index", range(len(VERIFIER_NODES) - 1))
    def test_route_targets_cover_every_reachable_node(self, index):
        targets = _route_targets(index, CALIBRATE)
        assert targets == NODE_NAMES[index + 1:] + [CALIBRATE]

    @pytest.mark.parametrize("index", range(len(VERIFIER_NODES) - 1))
    def test_router_only_returns_declared_targets(self, index):
        # LangGraph는 선언되지 않은 목적지를 반환하면 실행 시점에 실패한다.
        targets = set(_route_targets(index, CALIBRATE))
        for active in (["factual"], ["factual", "entity"], NODE_NAMES,
                       ["factual", "spatiotemporal", "visual"]):
            assert route(index, active) in targets


class TestBypassTriage:
    def test_activates_every_agent_when_an_image_exists(self):
        result = _bypass_triage({"image_paths": ["a.jpg"], "metrics": {}})
        assert result["active_agents"] == ["factual", "entity", "spatiotemporal", "visual"]
        assert result["metrics"]["active_agent_count"] == 4

    def test_still_requires_an_image_for_the_visual_agent(self):
        result = _bypass_triage({"image_paths": [], "metrics": {}})
        assert "visual" not in result["active_agents"]

    def test_reports_zero_triage_latency(self):
        # triage를 제거한 ablation이므로 분류 호출 비용이 없다.
        assert _bypass_triage({"image_paths": [], "metrics": {}})["metrics"]["triage_latency"] == 0.0


class TestAblations:
    def test_full_changes_nothing(self):
        assert ABLATIONS["full"] == {}

    def test_no_orchestration_disables_both_triage_and_calibrator(self):
        assert ABLATIONS["no_orchestration"] == {"use_triage": False, "use_calibrator": False}

    @pytest.mark.parametrize("name", ["no_gating", "no_triage", "no_calibrator"])
    def test_each_ablation_turns_off_exactly_one_component(self, name):
        assert len(ABLATIONS[name]) == 1
        assert all(value is False for value in ABLATIONS[name].values())


class TestInitialState:
    def test_carries_every_input_component(self):
        state = initial_state(
            claim="c", claim_id="1", post_text="P", ocr_text="O",
            image_paths=["i.jpg"], evidence_pool=["e"],
        )
        assert (state["claim"], state["post_text"], state["ocr_text"]) == ("c", "P", "O")
        assert state["image_paths"] == ["i.jpg"] and state["evidence_pool"] == ["e"]

    def test_defaults_are_empty_not_none(self):
        state = initial_state(claim="c")
        assert state["image_paths"] == [] and state["evidence_pool"] == []
        assert state["verdicts"] == {} and state["verdict"] == ""

    def test_does_not_share_mutable_defaults(self):
        first, second = initial_state(claim="a"), initial_state(claim="b")
        first["image_paths"].append("x")
        assert second["image_paths"] == []
