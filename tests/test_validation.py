"""
Tests for validation module.
"""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generators import hb_lfr
from src.validation.structural import (
    experiment_1_parameter_control,
    experiment_4_concentration,
)
from src.validation.statistical_tests import (
    test_monotonicity as monotonicity_test,
    bonferroni_correction,
    fdr_correction,
    compute_effect_size_and_ci,
)


class TestStructuralValidation:
    """Tests for structural validation experiments."""

    @pytest.fixture
    def small_params(self):
        """Small parameters for quick tests."""
        return {"n": 100, "mu": 0.3}

    def test_experiment_1_runs(self, small_params):
        """Test that experiment 1 runs without error."""
        results = experiment_1_parameter_control(
            generator_func=hb_lfr,
            generator_params=small_params,
            h_values=[0.0, 0.5, 1.0],
            n_samples=3,
            seed=42,
        )

        assert "h_values" in results
        assert "rho_mean" in results
        assert "monotonicity_test" in results

    def test_experiment_4_runs(self, small_params):
        """Test that experiment 4 runs without error."""
        results = experiment_4_concentration(
            generator_func=hb_lfr,
            generator_params=small_params,
            h_values=[0.0, 0.5],
            n_samples=5,
            seed=42,
        )

        assert "cv" in results
        assert "is_concentrated" in results


class TestStatisticalTests:
    """Tests for statistical testing functions."""

    def test_monotonicity_increasing(self):
        """Test monotonicity detection for increasing data."""
        x = [0.0, 0.5, 1.0]
        y = np.array([[1, 1.1, 0.9], [1.5, 1.6, 1.4], [2, 2.1, 1.9]])

        result = monotonicity_test(x, y)

        assert result["mean_increases"] is True
        assert result["spearman_r"] > 0

    def test_monotonicity_non_increasing(self):
        """Test monotonicity detection for non-increasing data."""
        x = [0.0, 0.5, 1.0]
        y = np.array([[2, 2.1, 1.9], [1.5, 1.6, 1.4], [1, 1.1, 0.9]])

        result = monotonicity_test(x, y)

        assert result["mean_increases"] is False

    def test_bonferroni_correction(self):
        """Test Bonferroni correction."""
        # With 5 tests and alpha=0.05, adjusted alpha = 0.01
        # So only p-values < 0.01 are significant
        p_values = [0.005, 0.02, 0.03, 0.04, 0.05]
        significant = bonferroni_correction(p_values, alpha=0.05)

        # Only first (0.005 < 0.01) should be significant after correction
        assert significant[0] is True
        assert all(not s for s in significant[1:])

    def test_fdr_correction(self):
        """Test FDR correction."""
        p_values = [0.001, 0.01, 0.02, 0.04, 0.05]
        significant, adjusted = fdr_correction(p_values, alpha=0.05)

        # More lenient than Bonferroni
        assert sum(significant) >= 1
        assert len(adjusted) == len(p_values)

    def test_effect_size_calculation(self):
        """Test effect size calculation."""
        group1 = [1, 2, 3, 4, 5]
        group2 = [3, 4, 5, 6, 7]

        result = compute_effect_size_and_ci(group1, group2, seed=42)

        assert "cohens_d" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "interpretation" in result

        # Effect size should be negative (group1 < group2)
        assert result["cohens_d"] < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
