"""Gated Evidence Retrieval 검증 (논문 3.3절)."""

import pytest

from prism.agents import retrieval_text, retrieval_visual

DOCS = [
    "Lauren Boebert owns Shooters Grill in Rifle, Colorado.",
    "The autopsy report attributes the death to a methamphetamine overdose.",
    "Local weather in Colorado was mild throughout the week.",
]


class TestBm25Search:
    def test_returns_the_most_relevant_document_first(self):
        assert retrieval_text.bm25_search(DOCS, "autopsy overdose report", top_n=1) == [DOCS[1]]

    def test_empty_pool_returns_empty_list(self):
        assert retrieval_text.bm25_search([], "anything") == []

    def test_never_returns_more_than_the_pool(self):
        assert len(retrieval_text.bm25_search(DOCS, "Colorado", top_n=10)) == len(DOCS)

    def test_matching_is_case_insensitive(self):
        assert retrieval_text.bm25_search(DOCS, "SHOOTERS GRILL", top_n=1) == [DOCS[0]]

    def test_uses_the_documented_bm25_parameters(self):
        assert (retrieval_text.BM25_K1, retrieval_text.BM25_B) == (1.5, 0.75)


class TestFormatEvidence:
    def test_labels_documents_as_e1_e2(self):
        formatted = retrieval_text.format_evidence(["first", "second"])
        assert formatted[0].startswith("[E1] ")
        assert formatted[1].startswith("[E2] ")

    def test_long_documents_are_truncated(self):
        formatted = retrieval_text.format_evidence(["x" * 50], max_chars=10)
        assert "x" * 10 in formatted[0] and "x" * 11 not in formatted[0]
        assert "truncated" in formatted[0]

    def test_short_documents_are_untouched(self):
        assert retrieval_text.format_evidence(["short"], max_chars=10) == ["[E1] short"]


class TestFallbackPool:
    def test_is_deterministic_across_calls(self, monkeypatch):
        monkeypatch.setattr(retrieval_text, "FALLBACK_POOL_SIZE", 5)
        index = {str(i): [f"doc {i}"] for i in range(50)}
        assert retrieval_text._fallback_pool(index) == retrieval_text._fallback_pool(index)

    def test_small_index_is_returned_whole(self):
        assert sorted(retrieval_text._fallback_pool({"1": ["a", "b"], "2": ["c"]})) == ["a", "b", "c"]


class TestTextualRetrievalAgent:
    @pytest.fixture
    def index(self, monkeypatch):
        data = {"42": DOCS}
        monkeypatch.setattr(retrieval_text, "claim_documents", lambda: data)
        return data

    def _state(self, claim, claim_id="42", **extra):
        return {"claim": claim, "claim_id": claim_id, "metrics": {}, **extra}

    def test_uses_documents_linked_to_the_claim_id(self, index):
        result = retrieval_text.textual_retrieval_agent(self._state("autopsy overdose"))
        assert len(result["text_evidence"]) == retrieval_text.TOP_K
        assert result["text_evidence"][0] == f"[E1] {DOCS[1]}"
        assert result["metrics"]["evidence_source"] == "claim_linked"

    def test_benchmark_supplied_pool_takes_precedence(self, index):
        # RW-Post처럼 벤치마크가 gold 증거를 직접 주는 경우다.
        state = self._state("anything", evidence_pool=["only this snippet"])
        result = retrieval_text.textual_retrieval_agent(state)
        assert result["text_evidence"] == ["[E1] only this snippet"]
        assert result["metrics"]["evidence_source"] == "state_pool"

    def test_unknown_claim_id_falls_back_to_the_corpus(self, index):
        result = retrieval_text.textual_retrieval_agent(self._state("Shooters Grill", "999"))
        assert result["text_evidence"]
        assert result["metrics"]["evidence_source"] == "corpus_fallback"

    def test_closed_book_returns_no_evidence(self, index):
        # 게이팅 제거 ablation. E_T 가 비어야 한다.
        result = retrieval_text.textual_retrieval_agent(self._state("autopsy"), gated=False)
        assert result["text_evidence"] == []
        assert result["metrics"]["text_retrieval_calls"] == 0

    def test_metrics_are_accumulated(self, index):
        state = self._state("Colorado")
        state["metrics"] = {"text_retrieval_calls": 2}
        result = retrieval_text.textual_retrieval_agent(state)
        assert result["metrics"]["text_retrieval_calls"] == 3
        assert result["metrics"]["retrieval_text_latency"] >= 0


class TestVisualGating:
    def test_mask_off_yields_no_image_evidence(self, tmp_path):
        image = tmp_path / "a.jpg"
        image.write_bytes(b"x")
        assert retrieval_visual.gate_images([str(image)], image_needed=False) == []

    def test_mask_on_passes_existing_files(self, tmp_path):
        image = tmp_path / "a.jpg"
        image.write_bytes(b"x")
        assert retrieval_visual.gate_images([str(image)], image_needed=True) == [str(image)]

    def test_missing_files_are_dropped(self):
        assert retrieval_visual.gate_images(["nope.jpg"], image_needed=True) == []

    def test_agent_records_metrics(self, tmp_path):
        image = tmp_path / "a.jpg"
        image.write_bytes(b"x")
        result = retrieval_visual.visual_retrieval_agent({
            "image_paths": [str(image)],
            "modality_mask": {"img": True},
            "metrics": {},
        })
        assert result["image_evidence"] == [str(image)]
        assert result["metrics"]["image_retrieval_calls"] == 1
