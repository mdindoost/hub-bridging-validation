"""
Tests for statistical tests module.
"""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.statistical_tests import (
    test_monotonicity as monotonicity_test,
    bonferroni_correction,
    fdr_correction,
    compute_effect_size_and_ci,
    two_sample_permutation_test,
)


class TestMonotonicity:
    """Tests for monotonicity testing."""

    def test_perfect_monotonic(self):
        """Test with perfectly monotonic data."""
        x = [1, 2, 3, 4, 5]
        y = np.array([[i, i, i] for i in x])

        result = monotonicity_test(x, y)
        assert result["mean_increases"] is True
        # Use approximate comparison for floating point
        assert abs(result["spearman_r"] - 1.0) < 1e-10

    def test_with_noise(self):
        """Test with noisy but monotonic data."""
        np.random.seed(42)
        x = [1, 2, 3, 4, 5]
        y = np.array([[i + np.random.normal(0, 0.1) for _ in range(10)] for i in x])

        result = monotonicity_test(x, y)
        assert result["spearman_r"] > 0.9


class TestMultipleTesting:
    """Tests for multiple testing corrections."""

    def test_bonferroni_all_significant(self):
        """Test Bonferroni when all are significant."""
        p_values = [0.001, 0.002, 0.003]
        significant = bonferroni_correction(p_values, alpha=0.05)
        assert all(significant)

    def test_bonferroni_none_significant(self):
        """Test Bonferroni when none are significant."""
        p_values = [0.5, 0.6, 0.7]
        significant = bonferroni_correction(p_values, alpha=0.05)
        assert not any(significant)

    def test_fdr_preserves_order(self):
        """Test that FDR adjusted p-values preserve order."""
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        _, adjusted = fdr_correction(p_values)

        # Adjusted should maintain relative ordering
        for i in range(len(adjusted) - 1):
            assert adjusted[i] <= adjusted[i + 1]


class TestEffectSize:
    """Tests for effect size calculations."""

    def test_large_effect(self):
        """Test detection of large effect."""
        group1 = [1, 2, 3, 4, 5]
        group2 = [6, 7, 8, 9, 10]

        result = compute_effect_size_and_ci(group1, group2, seed=42)
        assert result["interpretation"] == "large"

    def test_small_effect(self):
        """Test detection of small effect."""
        group1 = [1, 2, 3, 4, 5]
        group2 = [1.5, 2.5, 3.5, 4.5, 5.5]

        result = compute_effect_size_and_ci(group1, group2, seed=42)
        assert result["interpretation"] in ["negligible", "small"]

    def test_ci_contains_point_estimate(self):
        """Test that CI contains the point estimate."""
        group1 = np.random.normal(0, 1, 50).tolist()
        group2 = np.random.normal(1, 1, 50).tolist()

        result = compute_effect_size_and_ci(group1, group2, seed=42)
        assert result["ci_lower"] <= result["cohens_d"] <= result["ci_upper"]


class TestPermutationTest:
    """Tests for permutation test."""

    def test_same_distribution(self):
        """Test permutation test with same distribution."""
        np.random.seed(42)
        group1 = np.random.normal(0, 1, 50).tolist()
        group2 = np.random.normal(0, 1, 50).tolist()

        result = two_sample_permutation_test(group1, group2, seed=42)
        # Should have high p-value
        assert result["p_value"] > 0.05

    def test_different_distributions(self):
        """Test permutation test with different distributions."""
        np.random.seed(42)
        group1 = np.random.normal(0, 1, 50).tolist()
        group2 = np.random.normal(2, 1, 50).tolist()

        result = two_sample_permutation_test(group1, group2, seed=42)
        # Should have low p-value
        assert result["p_value"] < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
