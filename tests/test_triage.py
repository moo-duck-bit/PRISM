"""S(m) 계산 검증 (논문 3.2절).

    S(m) = {f} u {e | m_ent} u {st | m_time} u {v | m_img and |I| >= 1}
"""

import pytest

from prism.agents.triage import ENTITY, FACTUAL, SPATIOTEMPORAL, VISUAL, active_agents

ALL_OFF = {"img": False, "ent": False, "time": False}
ALL_ON = {"img": True, "ent": True, "time": True}


def test_factual_is_always_active():
    assert active_agents(ALL_OFF, has_image=False) == [FACTUAL]


def test_entity_follows_its_mask_bit():
    assert ENTITY in active_agents({**ALL_OFF, "ent": True}, has_image=False)
    assert ENTITY not in active_agents(ALL_OFF, has_image=False)


def test_spatiotemporal_follows_its_mask_bit():
    assert SPATIOTEMPORAL in active_agents({**ALL_OFF, "time": True}, has_image=False)


def test_visual_needs_both_the_mask_and_an_actual_image():
    # 마스크만 켜져 있고 이미지가 없으면 시각 검증기는 돌지 않는다.
    assert VISUAL not in active_agents({**ALL_OFF, "img": True}, has_image=False)
    assert VISUAL in active_agents({**ALL_OFF, "img": True}, has_image=True)


def test_image_without_the_mask_does_not_activate_visual():
    assert VISUAL not in active_agents(ALL_OFF, has_image=True)


def test_full_mask_activates_all_four():
    assert active_agents(ALL_ON, has_image=True) == [
        FACTUAL, ENTITY, SPATIOTEMPORAL, VISUAL
    ]


@pytest.mark.parametrize("mask,has_image,expected", [
    ({"img": True, "ent": True, "time": False}, True, 3),   # 논문 Figure 1의 사례
    ({"img": False, "ent": True, "time": True}, False, 3),
    ({"img": False, "ent": False, "time": False}, False, 1),
])
def test_active_counts(mask, has_image, expected):
    assert len(active_agents(mask, has_image)) == expected


def test_order_matches_the_graph_execution_order():
    # 라우터가 이 순서대로 다음 활성 노드를 찾으므로 순서가 바뀌면 그래프가 어긋난다.
    agents = active_agents(ALL_ON, has_image=True)
    assert agents == sorted(agents, key=[FACTUAL, ENTITY, SPATIOTEMPORAL, VISUAL].index)
