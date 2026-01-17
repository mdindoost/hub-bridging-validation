"""
Tests for metrics module.

Comprehensive tests for hub-bridging metrics, network properties,
and distance metrics.
"""

import pytest
import networkx as nx
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.hub_bridging import (
    compute_hub_bridging_ratio,
    compute_dspar_separation,
    compute_dspar_score,
    classify_edges_by_hub_bridging,
    _parse_communities,
    _partition_edges,
)
from src.metrics.network_properties import (
    comprehensive_network_properties,
    compute_degree_distribution_stats,
    compute_participation_coefficient,
    compute_within_module_degree,
)
from src.metrics.distance_metrics import (
    maximum_mean_discrepancy,
    wasserstein_distance_1d,
    ks_distance,
)


class TestDSparScore:
    """Tests for DSpar score computation."""

    def test_dspar_score_basic(self):
        """Test DSpar score computation."""
        degrees = {0: 10, 1: 5}
        score = compute_dspar_score(0, 1, degrees)
        expected = 1/10 + 1/5  # 0.1 + 0.2 = 0.3
        assert abs(score - expected) < 1e-10

    def test_dspar_score_symmetric(self):
        """Test that DSpar score is symmetric."""
        degrees = {0: 10, 1: 5, 2: 2}
        assert compute_dspar_score(0, 1, degrees) == compute_dspar_score(1, 0, degrees)

    def test_dspar_score_high_low_degree(self):
        """Test DSpar score for high vs low degree nodes."""
        degrees = {0: 100, 1: 100, 2: 2, 3: 2}

        # High-degree edge: low DSpar score
        high_deg_score = compute_dspar_score(0, 1, degrees)  # 1/100 + 1/100 = 0.02

        # Low-degree edge: high DSpar score
        low_deg_score = compute_dspar_score(2, 3, degrees)  # 1/2 + 1/2 = 1.0

        assert low_deg_score > high_deg_score


class TestParseCommunities:
    """Tests for community parsing."""

    def test_parse_communities_list_of_sets(self):
        """Test parsing list of sets format."""
        communities_sets = [{0, 1, 2}, {3, 4, 5}]
        parsed = _parse_communities(communities_sets)
        assert parsed == {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    def test_parse_communities_dict(self):
        """Test parsing dict format."""
        communities_dict = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
        parsed = _parse_communities(communities_dict)
        assert parsed == communities_dict

    def test_parse_communities_list_of_lists(self):
        """Test parsing list of lists format."""
        communities_lists = [[0, 1, 2], [3, 4, 5]]
        parsed = _parse_communities(communities_lists)
        assert parsed == {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}

    def test_parse_communities_preserves_string_labels(self):
        """Test that string community labels are preserved."""
        communities_dict = {0: 'A', 1: 'A', 2: 'B', 3: 'B'}
        parsed = _parse_communities(communities_dict)
        assert parsed[0] == 'A'
        assert parsed[2] == 'B'


class TestPartitionEdges:
    """Tests for edge partitioning."""

    def test_partition_edges_basic(self):
        """Test basic edge partitioning."""
        G = nx.Graph([(0, 1), (1, 2), (2, 3)])
        node_to_comm = {0: 0, 1: 0, 2: 1, 3: 1}
        E_intra, E_inter = _partition_edges(G, node_to_comm)

        # Edge (0,1) is intra (both in comm 0)
        # Edge (1,2) is inter (comm 0 to comm 1)
        # Edge (2,3) is intra (both in comm 1)
        assert len(E_intra) == 2
        assert len(E_inter) == 1

    def test_partition_edges_all_intra(self):
        """Test when all edges are intra-community."""
        G = nx.Graph([(0, 1), (1, 2), (0, 2)])
        node_to_comm = {0: 0, 1: 0, 2: 0}
        E_intra, E_inter = _partition_edges(G, node_to_comm)

        assert len(E_intra) == 3
        assert len(E_inter) == 0

    def test_partition_edges_all_inter(self):
        """Test when all edges are inter-community."""
        G = nx.Graph([(0, 1), (0, 2), (1, 2)])
        node_to_comm = {0: 0, 1: 1, 2: 2}
        E_intra, E_inter = _partition_edges(G, node_to_comm)

        assert len(E_intra) == 0
        assert len(E_inter) == 3


class TestHubBridgingRatio:
    """Tests for hub-bridging ratio computation."""

    @pytest.fixture
    def karate_graph(self):
        """Create karate club graph with communities."""
        G = nx.karate_club_graph()
        communities = {n: 0 if n < 17 else 1 for n in G.nodes()}
        return G, communities

    def test_hub_bridging_ratio_karate(self, karate_graph):
        """Test hub-bridging ratio on karate club."""
        G, communities = karate_graph
        rho = compute_hub_bridging_ratio(G, communities)

        # Should be positive
        assert rho > 0

        # Should be a reasonable value
        assert 0.1 < rho < 100

    def test_hub_bridging_simple_graph(self):
        """Test hub-bridging on simple constructed graph."""
        # Create graph with known hub-bridging structure
        G = nx.Graph()

        # Community 1: nodes 0-2, Community 2: nodes 3-5
        # High-degree hubs: 0 (d=3), 3 (d=3)
        # Low-degree nodes: 1,2,4,5 (d=2 each)

        # Intra-edges (connect low-degree nodes within communities)
        G.add_edges_from([(1, 2), (4, 5)])  # d_u*d_v = 4 each

        # Inter-edges (connect hubs across communities)
        G.add_edges_from([(0, 3)])  # Will have high d_u*d_v

        # Add edges to hubs to get desired degrees
        G.add_edges_from([(0, 1), (0, 2), (3, 4), (3, 5)])

        communities = [{0, 1, 2}, {3, 4, 5}]

        rho = compute_hub_bridging_ratio(G, communities)

        # Inter: (0,3) with degrees 3,3 -> product = 9
        # Intra: (1,2) d=2,2 -> 4; (4,5) d=2,2 -> 4
        #        (0,1) d=3,2 -> 6; (0,2) d=3,2 -> 6
        #        (3,4) d=3,2 -> 6; (3,5) d=3,2 -> 6
        # mean_inter = 9
        # mean_intra = (4+4+6+6+6+6)/6 = 32/6 = 5.33
        # Expected rho ≈ 9 / 5.33 ≈ 1.69

        assert rho > 1.0, "Should have hub-bridging (rho > 1)"

    def test_hub_bridging_uniform_mixing(self):
        """Test that uniform mixing gives rho close to 1."""
        # Create graph where inter and intra edges have similar degree products
        G = nx.Graph()

        # Two communities with mixed degree nodes
        # All nodes have similar degrees (complete within communities)
        edges = [
            # Intra edges
            (0, 1), (0, 2), (1, 2),  # Community 1
            (3, 4), (3, 5), (4, 5),  # Community 2
            # Inter edges (symmetric mixing)
            (0, 3), (1, 4), (2, 5)
        ]
        G.add_edges_from(edges)

        communities = [{0, 1, 2}, {3, 4, 5}]
        rho = compute_hub_bridging_ratio(G, communities)

        # All nodes have degree 3, so all edge products are 9
        # Should be exactly 1.0
        assert 0.9 < rho < 1.1, f"Expected rho ≈ 1, got {rho}"

    def test_hub_bridging_accepts_list_of_sets(self):
        """Test that list of sets format works."""
        G = nx.karate_club_graph()
        communities = [{n for n in range(17)}, {n for n in range(17, 34)}]
        rho = compute_hub_bridging_ratio(G, communities)
        assert rho > 0

    def test_hub_bridging_accepts_dict(self):
        """Test that dict format works."""
        G = nx.karate_club_graph()
        communities = {n: 0 if n < 17 else 1 for n in G.nodes()}
        rho = compute_hub_bridging_ratio(G, communities)
        assert rho > 0

    def test_hub_bridging_no_inter_edges_raises(self):
        """Test that no inter-edges raises ValueError."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])

        # All nodes in same community (no inter-edges)
        communities_single = [{0, 1, 2}]

        with pytest.raises(ValueError, match="no inter-community edges"):
            compute_hub_bridging_ratio(G, communities_single)

    def test_hub_bridging_no_intra_edges_raises(self):
        """Test that no intra-edges raises ValueError."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])

        # All nodes in different communities (no intra-edges)
        communities_all_different = [{0}, {1}, {2}]

        with pytest.raises(ValueError, match="no intra-community edges"):
            compute_hub_bridging_ratio(G, communities_all_different)


class TestDSparSeparation:
    """Tests for DSpar separation metric."""

    def test_dspar_separation_basic(self):
        """Test DSpar separation on karate club."""
        G = nx.karate_club_graph()
        communities = {n: 0 if n < 17 else 1 for n in G.nodes()}
        delta = compute_dspar_separation(G, communities)

        # Should return a finite value
        assert np.isfinite(delta)

    def test_dspar_separation_inversely_related(self):
        """Test that delta and rho are inversely related."""
        # Use graph with hub-bridging (inter-edges connect hubs)
        G = nx.Graph()
        G.add_edges_from([(1, 2), (4, 5), (0, 3), (0, 1), (0, 2), (3, 4), (3, 5)])
        communities = [{0, 1, 2}, {3, 4, 5}]

        rho = compute_hub_bridging_ratio(G, communities)
        delta = compute_dspar_separation(G, communities)

        # If rho > 1 (hub-bridging), inter-edges connect hubs (low DSpar)
        # So intra-edges have higher DSpar -> delta = mu_intra - mu_inter > 0
        if rho > 1.0:
            assert delta > 0.0, f"delta should be positive when rho > 1 (rho={rho}, delta={delta})"

    def test_dspar_separation_no_inter_edges_raises(self):
        """Test that no inter-edges raises ValueError."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        communities_single = [{0, 1, 2}]

        with pytest.raises(ValueError, match="no inter-community edges"):
            compute_dspar_separation(G, communities_single)


class TestClassifyEdges:
    """Tests for edge classification."""

    def test_classify_edges_complete(self, ):
        """Test that all edges are classified."""
        G = nx.karate_club_graph()
        communities = {n: 0 if n < 17 else 1 for n in G.nodes()}
        inter, intra = classify_edges_by_hub_bridging(G, communities)

        # All edges should be classified
        assert len(inter) + len(intra) == G.number_of_edges()

        # Both types should exist
        assert len(inter) > 0
        assert len(intra) > 0


class TestNetworkProperties:
    """Tests for network property computations."""

    @pytest.fixture
    def test_graph(self):
        """Create test graph."""
        G = nx.barabasi_albert_graph(100, 3, seed=42)
        return G

    def test_degree_distribution_stats(self, test_graph):
        """Test degree distribution statistics."""
        stats = compute_degree_distribution_stats(test_graph)

        assert "mean" in stats
        assert "std" in stats
        assert "max" in stats
        assert stats["mean"] > 0
        assert stats["max"] >= stats["mean"]

    def test_participation_coefficient(self):
        """Test participation coefficient."""
        G = nx.karate_club_graph()
        communities = {n: 0 if n < 17 else 1 for n in G.nodes()}

        P = compute_participation_coefficient(G, communities)

        assert len(P) == G.number_of_nodes()
        assert all(0 <= p <= 1 for p in P)

    def test_within_module_degree(self):
        """Test within-module degree z-score."""
        G = nx.karate_club_graph()
        communities = {n: 0 if n < 17 else 1 for n in G.nodes()}

        z = compute_within_module_degree(G, communities)

        assert len(z) == G.number_of_nodes()
        # Z-scores should be centered around 0
        assert abs(np.mean(z)) < 1

    def test_comprehensive_properties(self):
        """Test comprehensive property computation."""
        G = nx.karate_club_graph()
        communities = {n: 0 if n < 17 else 1 for n in G.nodes()}

        props = comprehensive_network_properties(G, communities, compute_expensive=False)

        assert "basic" in props
        assert "degree" in props
        assert "clustering" in props
        assert props["basic"]["n_nodes"] == 34


class TestDistanceMetrics:
    """Tests for distance metrics."""

    def test_mmd_same_distribution(self):
        """Test MMD between identical distributions."""
        np.random.seed(42)
        X = np.random.normal(0, 1, 100)
        Y = np.random.normal(0, 1, 100)

        mmd = maximum_mean_discrepancy(X, Y)

        # Should be small for same distribution
        assert mmd < 0.5

    def test_mmd_different_distributions(self):
        """Test MMD between different distributions."""
        np.random.seed(42)
        X = np.random.normal(0, 1, 100)
        Y = np.random.normal(5, 1, 100)

        mmd = maximum_mean_discrepancy(X, Y)

        # Should be larger for different distributions
        assert mmd > 0.1

    def test_wasserstein_distance(self):
        """Test Wasserstein distance."""
        X = np.array([1, 2, 3, 4, 5])
        Y = np.array([2, 3, 4, 5, 6])

        w = wasserstein_distance_1d(X, Y)

        # Should be approximately 1 (shift of 1)
        assert abs(w - 1.0) < 0.1

    def test_ks_distance_same(self):
        """Test KS distance for same distribution."""
        np.random.seed(42)
        X = np.random.normal(0, 1, 100)
        Y = np.random.normal(0, 1, 100)

        ks, p = ks_distance(X, Y)

        # Should have high p-value
        assert p > 0.01

    def test_ks_distance_different(self):
        """Test KS distance for different distributions."""
        np.random.seed(42)
        X = np.random.normal(0, 1, 100)
        Y = np.random.normal(3, 1, 100)

        ks, p = ks_distance(X, Y)

        # Should have low p-value
        assert p < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
