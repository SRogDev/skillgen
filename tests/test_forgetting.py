"""Integrity tests for the local forgetting-check question set (Phase 4).

The LOCAL_QUESTIONS set is an original local set (NOT ARC); these tests
only certify its shape so the loglikelihood harness can consume it.
"""

from skillgen.evaluation.forgetting import LOCAL_QUESTIONS


def test_local_questions_shape():
    assert len(LOCAL_QUESTIONS) == 20
    topics = set()
    for q in LOCAL_QUESTIONS:
        assert len(q["options"]) == 4
        assert 0 <= q["answer"] < 4
        assert len(set(q["options"])) == 4
        topics.add(q["topic"])
    assert topics == {"science", "history", "math", "geography", "language"}
