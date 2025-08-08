"""Tests for optional Numba functionality."""

import pytest

import ranx
from ranx.config import reset_numba_config


class TestNumbaOptional:
    """Test that ranx works with Numba both enabled and disabled."""

    def setup_method(self):
        """Reset configuration before each test."""
        reset_numba_config()

    def test_precision_with_numba_enabled(self):
        """Test precision calculation with Numba enabled."""
        ranx.set_numba_enabled(True)

        qrels_dict = {"q1": {"d1": 1, "d2": 1, "d3": 0}}
        run_dict = {"q1": {"d1": 0.9, "d2": 0.8, "d3": 0.7}}

        qrels = ranx.Qrels.from_dict(qrels_dict)
        run = ranx.Run.from_dict(run_dict)

        result = ranx.evaluate(qrels, run, ["precision@2"])

        # Should get 2/2 = 1.0 precision (both top-2 docs are relevant)
        assert result == pytest.approx(1.0)

    def test_precision_with_numba_disabled(self):
        """Test precision calculation with Numba disabled."""
        ranx.set_numba_enabled(False)

        qrels_dict = {"q1": {"d1": 1, "d2": 1, "d3": 0}}
        run_dict = {"q1": {"d1": 0.9, "d2": 0.8, "d3": 0.7}}

        qrels = ranx.Qrels.from_dict(qrels_dict)
        run = ranx.Run.from_dict(run_dict)

        result = ranx.evaluate(qrels, run, ["precision@2"])

        # Should get the same result as with Numba enabled
        assert result == pytest.approx(1.0)

    def test_recall_with_numba_disabled(self):
        """Test recall calculation with Numba disabled."""
        ranx.set_numba_enabled(False)

        qrels_dict = {"q1": {"d1": 1, "d2": 1, "d3": 1}}  # 3 relevant docs
        run_dict = {"q1": {"d1": 0.9, "d2": 0.8, "d3": 0.7}}

        qrels = ranx.Qrels.from_dict(qrels_dict)
        run = ranx.Run.from_dict(run_dict)

        result = ranx.evaluate(qrels, run, ["recall@2"])

        # Should get 2/3 ≈ 0.667 recall (found 2 out of 3 relevant docs in top-2)
        assert result == pytest.approx(2 / 3)

    def test_hits_with_numba_disabled(self):
        """Test hits calculation with Numba disabled."""
        ranx.set_numba_enabled(False)

        qrels_dict = {"q1": {"d1": 1, "d2": 1, "d3": 0}}
        run_dict = {"q1": {"d1": 0.9, "d2": 0.8, "d3": 0.7}}

        qrels = ranx.Qrels.from_dict(qrels_dict)
        run = ranx.Run.from_dict(run_dict)

        result = ranx.evaluate(qrels, run, ["hits@2"])

        # Should get 2 hits (both top-2 docs are relevant)
        assert result == pytest.approx(2.0)

    def test_multiple_queries_with_numba_disabled(self):
        """Test multiple queries with Numba disabled."""
        ranx.set_numba_enabled(False)

        qrels_dict = {"q1": {"d1": 1, "d2": 1}, "q2": {"d3": 1, "d4": 1, "d5": 1}}
        run_dict = {
            "q1": {"d1": 0.9, "d2": 0.8, "d6": 0.7},
            "q2": {"d3": 0.95, "d4": 0.85, "d5": 0.75},
        }

        qrels = ranx.Qrels.from_dict(qrels_dict)
        run = ranx.Run.from_dict(run_dict)

        result = ranx.evaluate(qrels, run, ["precision@2", "recall@2"])

        # q1: precision@2 = 2/2 = 1.0, recall@2 = 2/2 = 1.0
        # q2: precision@2 = 2/2 = 1.0, recall@2 = 2/3 = 0.667
        # Mean precision@2 = (1.0 + 1.0) / 2 = 1.0
        # Mean recall@2 = (1.0 + 0.667) / 2 = 0.833
        assert result["precision@2"] == pytest.approx(1.0)
        assert result["recall@2"] == pytest.approx(5 / 6, abs=1e-3)

    def test_configuration_via_function(self):
        """Test configuration via set_numba_enabled function."""
        # Test enabling
        ranx.set_numba_enabled(True)
        assert ranx.use_numba() is True

        # Test disabling
        ranx.set_numba_enabled(False)
        assert ranx.use_numba() is False

    def test_same_results_numba_enabled_disabled(self):
        """Test that results are the same with Numba enabled vs disabled."""
        qrels_dict = {"q1": {"d1": 2, "d2": 1, "d3": 0}}
        run_dict = {"q1": {"d1": 0.9, "d2": 0.8, "d3": 0.7, "d4": 0.6}}

        qrels = ranx.Qrels.from_dict(qrels_dict)
        run = ranx.Run.from_dict(run_dict)

        # Test with Numba enabled
        ranx.set_numba_enabled(True)
        result_numba = ranx.evaluate(qrels, run, ["precision@3", "recall@3", "hits@3"])

        # Test with Numba disabled
        ranx.set_numba_enabled(False)
        result_no_numba = ranx.evaluate(
            qrels, run, ["precision@3", "recall@3", "hits@3"]
        )

        # Results should be identical
        for metric in ["precision@3", "recall@3", "hits@3"]:
            assert result_numba[metric] == pytest.approx(result_no_numba[metric])
