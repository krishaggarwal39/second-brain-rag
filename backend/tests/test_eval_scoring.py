"""R-35: evaluation scoring must not auto-pass off-topic queries."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "evaluation"))
custom_eval = pytest.importorskip("custom_eval")


def test_off_topic_with_relevant_looking_context_fails():
    retrieved = [{"text": "a joke about finance", "rerank_score": 3.2}]
    assert custom_eval.out_of_scope_passes(retrieved, threshold=-5.0) is False


def test_off_topic_with_only_irrelevant_context_passes():
    retrieved = [{"text": "x", "rerank_score": -9.0}, {"text": "y", "rerank_score": -11.0}]
    assert custom_eval.out_of_scope_passes(retrieved, threshold=-5.0) is True
    assert custom_eval.out_of_scope_passes([], threshold=-5.0) is True


def test_in_scope_empty_retrieval_is_not_a_pass():
    assert custom_eval.compute_source_precision([], "doc.pdf") == 0.0
    assert custom_eval.compute_keyword_recall([], ["revenue"]) == 0.0


def test_in_scope_queries_require_labels():
    with pytest.raises(ValueError):
        custom_eval.compute_keyword_recall(["x"], [])
    with pytest.raises(ValueError):
        custom_eval.compute_source_precision([{"filename": "a"}], None)
