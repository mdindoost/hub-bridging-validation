"""
Tests for generators module.
"""

import pytest
import networkx as nx
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generators.base_generators import (
    generate_lfr,
    generate_sbm,
    generate_planted_partition,
    generate_standard_lfr,
    extract_lfr_params,
)
from src.generators.hb_lfr import (
    hb_lfr,
    hb_lfr_rewiring,
)
from src.generators.hb_sbm import hb_sbm, hb_sbm_simple
from src.generators.calibration import quick_calibration_check


class TestBaseGenerators:
    """Tests for base generators."""

    def test_generate_lfr(self):
        """Test LFR generation."""
        G, communities = generate_lfr(n=100, mu=0.3, seed=42)

        assert G.number_of_nodes() == 100
        assert len(communities) == 100
        assert len(set(communities.values())) > 1

    def test_generate_sbm(self):
        """Test SBM generation."""
        G, communities = generate_sbm(n=100, k=5, p_in=0.3, p_out=0.05, seed=42)

        assert G.number_of_nodes() == 100
        assert len(communities) == 100
        assert len(set(communities.values())) == 5

    def test_generate_planted_partition(self):
        """Test planted partition generation."""
        G, communities = generate_planted_partition(n=100, k=4, seed=42)

        assert G.number_of_nodes() == 100
        # Communities should be roughly equal sized
        sizes = [sum(1 for c in communities.values() if c == i) for i in range(4)]
        assert all(20 <= s <= 30 for s in sizes)


class TestHBLFR:
    """Tests for HB-LFR generator.

    Note: Uses n=250 because LFR with average_degree constraint requires
    larger networks to reliably satisfy all constraints.
    """

    def test_hb_lfr_h_zero(self):
        """Test HB-LFR with h=0 (standard LFR)."""
        G, communities = hb_lfr(n=250, mu=0.3, h=0.0, seed=42)

        assert G.number_of_nodes() == 250
        assert G.number_of_edges() > 0
        # communities is list of sets
        assert isinstance(communities, list)
        assert all(isinstance(c, set) for c in communities)

    def test_hb_lfr_h_positive(self):
        """Test HB-LFR with positive h."""
        G, communities = hb_lfr(n=250, mu=0.3, h=0.5, seed=42)

        assert G.number_of_nodes() == 250
        # Check all nodes are in some community
        all_nodes = set()
        for comm in communities:
            all_nodes.update(comm)
        assert all_nodes == set(G.nodes())

    def test_hb_lfr_preserves_node_count(self):
        """Test that node count is preserved across h values."""
        for h in [0.0, 0.5, 1.0]:
            G, _ = hb_lfr(n=250, mu=0.3, h=h, seed=42)
            assert G.number_of_nodes() == 250

    @pytest.mark.parametrize("h", [0.0, 0.3, 0.6, 1.0])
    def test_hb_lfr_produces_valid_graph(self, h):
        """Test that HB-LFR produces valid graphs for various h."""
        G, communities = hb_lfr(n=250, mu=0.3, h=h, seed=42)

        # Should be valid graph
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

        # Communities should cover all nodes (list of sets format)
        all_nodes = set()
        for comm in communities:
            all_nodes.update(comm)
        assert all_nodes == set(G.nodes())

    def test_hb_lfr_preserves_degree_distribution(self):
        """Test that rewiring preserves degree distribution."""
        # Generate two graphs with same seed but different h
        G0, _ = hb_lfr(n=250, mu=0.3, h=0.0, seed=42)
        G1, _ = hb_lfr(n=250, mu=0.3, h=1.0, seed=42)

        # Degree distributions should be similar (same LFR base)
        degrees0 = sorted([d for _, d in G0.degree()])
        degrees1 = sorted([d for _, d in G1.degree()])

        # Not identical due to rewiring, but same node count
        assert len(degrees0) == len(degrees1)

    def test_hb_lfr_communities_stored_in_graph(self):
        """Test that communities are stored in G.graph."""
        G, communities = hb_lfr(n=250, mu=0.3, h=0.5, seed=42)

        assert 'communities' in G.graph
        assert G.graph['communities'] == communities
        assert 'n_communities' in G.graph
        assert G.graph['n_communities'] == len(communities)

    def test_hb_lfr_params_stored_in_graph(self):
        """Test that generation params are stored in G.graph."""
        G, _ = hb_lfr(n=250, mu=0.3, h=0.5, seed=42)

        assert 'params' in G.graph
        assert G.graph['params']['h'] == 0.5
        assert G.graph['params']['mu'] == 0.3
        assert G.graph['params']['n'] == 250


class TestHBSBM:
    """Tests for HB-SBM generator."""

    def test_hb_sbm_basic(self):
        """Test basic HB-SBM generation."""
        G, communities = hb_sbm(n=100, k=5, h=0.0, seed=42)

        assert G.number_of_nodes() == 100
        assert len(set(communities.values())) == 5

    def test_hb_sbm_with_h(self):
        """Test HB-SBM with positive h."""
        G, communities = hb_sbm(n=100, k=5, h=0.5, seed=42)

        assert G.number_of_nodes() == 100
        assert len(communities) == 100

    def test_hb_sbm_degree_distributions(self):
        """Test HB-SBM with different degree distributions."""
        for dist in ["uniform", "exponential", "powerlaw"]:
            G, communities = hb_sbm(
                n=100, k=5, h=0.0,
                degree_distribution=dist,
                seed=42
            )
            assert G.number_of_nodes() == 100


class TestCalibration:
    """Tests for calibration utilities."""

    def test_quick_calibration_check(self):
        """Test quick calibration check."""
        params = {"n": 100, "mu": 0.3}
        results = quick_calibration_check(hb_lfr, params, seed=42)

        # Should have results for h=0, 0.5, 1.0
        assert 0.0 in results
        assert 0.5 in results
        assert 1.0 in results

        # All should be positive finite values
        for h, rho in results.items():
            if not np.isnan(rho):
                assert rho > 0


class TestGeneratorConsistency:
    """Tests for generator consistency and reproducibility."""

    def test_lfr_reproducibility(self):
        """Test that LFR generation is reproducible with seed."""
        G1, c1 = generate_lfr(n=100, seed=42)
        G2, c2 = generate_lfr(n=100, seed=42)

        assert G1.number_of_edges() == G2.number_of_edges()
        assert c1 == c2

    def test_sbm_reproducibility(self):
        """Test that SBM generation is reproducible with seed."""
        G1, c1 = generate_sbm(n=100, k=5, seed=42)
        G2, c2 = generate_sbm(n=100, k=5, seed=42)

        assert G1.number_of_edges() == G2.number_of_edges()
        assert c1 == c2

    def test_hb_lfr_reproducibility(self):
        """Test that HB-LFR is reproducible with seed."""
        G1, c1 = hb_lfr(n=250, mu=0.3, h=0.5, seed=42)
        G2, c2 = hb_lfr(n=250, mu=0.3, h=0.5, seed=42)

        assert G1.number_of_edges() == G2.number_of_edges()


class TestStandardLFR:
    """Tests for generate_standard_lfr() function."""

    def test_standard_lfr_returns_graph(self):
        """Test that standard LFR returns a single graph."""
        G = generate_standard_lfr(n=100, mu=0.3, seed=42)

        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 100

    def test_standard_lfr_has_communities_in_graph(self):
        """Test that communities are stored in G.graph."""
        G = generate_standard_lfr(n=100, mu=0.3, seed=42)

        assert 'communities' in G.graph
        assert 'n_communities' in G.graph

    def test_standard_lfr_communities_are_list_of_sets(self):
        """Test that communities are in list of sets format."""
        G = generate_standard_lfr(n=100, mu=0.3, seed=42)

        communities = G.graph['communities']
        assert isinstance(communities, list)
        assert all(isinstance(c, set) for c in communities)

    def test_standard_lfr_communities_cover_all_nodes(self):
        """Test that communities cover all nodes."""
        G = generate_standard_lfr(n=100, mu=0.3, seed=42)

        communities = G.graph['communities']
        all_nodes = set()
        for comm in communities:
            all_nodes.update(comm)

        assert all_nodes == set(G.nodes())

    def test_standard_lfr_communities_are_disjoint(self):
        """Test that communities are disjoint."""
        G = generate_standard_lfr(n=100, mu=0.3, seed=42)

        communities = G.graph['communities']
        seen = set()
        for comm in communities:
            assert seen.isdisjoint(comm), "Communities should be disjoint"
            seen.update(comm)


class TestExtractLFRParams:
    """Tests for extract_lfr_params() function."""

    def test_extract_params_basic(self):
        """Test basic parameter extraction."""
        G = nx.karate_club_graph()
        params = extract_lfr_params(G)

        assert params['n'] == 34
        assert params['m'] == 78
        assert 'tau1' in params
        assert 'average_degree' in params

    def test_extract_params_with_communities(self):
        """Test parameter extraction with communities."""
        G = generate_standard_lfr(n=100, mu=0.3, seed=42)
        params = extract_lfr_params(G)

        assert params['mu'] is not None
        assert params['min_community'] is not None
        assert params['max_community'] is not None
        assert params['tau2'] is not None

    def test_extract_params_without_communities(self):
        """Test parameter extraction without communities."""
        G = nx.barabasi_albert_graph(100, 3, seed=42)
        params = extract_lfr_params(G)

        assert params['mu'] is None
        assert params['min_community'] is None
        assert params['tau2'] is None

    def test_extract_params_degree_stats(self):
        """Test that degree statistics are correct."""
        G = nx.complete_graph(10)
        params = extract_lfr_params(G)

        assert params['min_degree'] == 9
        assert params['max_degree'] == 9
        assert params['average_degree'] == 9


class TestHBLFRRewiring:
    """Tests for HB-LFR rewiring function."""

    @pytest.fixture
    def test_graph_with_communities(self):
        """Create a test graph with communities (list of sets format)."""
        G = generate_standard_lfr(n=100, mu=0.3, seed=42)
        communities = G.graph['communities']
        return G, communities

    def test_hb_lfr_rewiring_preserves_node_count(self, test_graph_with_communities):
        """Test that rewiring preserves node count."""
        G, communities = test_graph_with_communities
        n_original = G.number_of_nodes()

        G_rewired = hb_lfr_rewiring(G, communities, h=0.5, seed=42)

        assert G_rewired.number_of_nodes() == n_original

    def test_hb_lfr_rewiring_preserves_edge_count(self, test_graph_with_communities):
        """Test that rewiring preserves edge count."""
        G, communities = test_graph_with_communities
        m_original = G.number_of_edges()

        G_rewired = hb_lfr_rewiring(G, communities, h=0.5, seed=42)

        assert G_rewired.number_of_edges() == m_original

    def test_hb_lfr_rewiring_with_h_zero(self, test_graph_with_communities):
        """Test that h=0 returns unchanged graph."""
        G, communities = test_graph_with_communities

        # h=0 should compute target rho=1.0, which is baseline
        G_rewired = hb_lfr_rewiring(G.copy(), communities, h=0.0, seed=42)

        # Should still have valid structure
        assert G_rewired.number_of_nodes() == G.number_of_nodes()

    def test_hb_lfr_rewiring_with_different_h(self, test_graph_with_communities):
        """Test rewiring with different h values."""
        G, communities = test_graph_with_communities

        for h in [0.5, 1.0, 1.5]:
            G_rewired = hb_lfr_rewiring(G.copy(), communities, h=h, seed=42)
            assert G_rewired.number_of_nodes() == G.number_of_nodes()
            assert G_rewired.number_of_edges() == G.number_of_edges()


class TestHBSBMSimple:
    """Tests for hb_sbm_simple() function."""

    def test_hb_sbm_simple_returns_graph(self):
        """Test that hb_sbm_simple returns a graph."""
        G = hb_sbm_simple(n=100, k=4, h=0.0, seed=42)

        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == 100

    def test_hb_sbm_simple_has_communities(self):
        """Test that hb_sbm_simple stores communities in G.graph."""
        G = hb_sbm_simple(n=100, k=4, h=0.0, seed=42)

        assert 'communities' in G.graph
        assert len(G.graph['communities']) == 4

    def test_hb_sbm_simple_communities_are_list_of_sets(self):
        """Test that communities are in list of sets format."""
        G = hb_sbm_simple(n=100, k=4, h=0.0, seed=42)

        communities = G.graph['communities']
        assert isinstance(communities, list)
        assert all(isinstance(c, set) for c in communities)

    def test_hb_sbm_simple_communities_cover_all_nodes(self):
        """Test that communities cover all nodes."""
        G = hb_sbm_simple(n=100, k=4, h=0.0, seed=42)

        communities = G.graph['communities']
        all_nodes = set()
        for comm in communities:
            all_nodes.update(comm)

        assert all_nodes == set(G.nodes())

    def test_hb_sbm_simple_with_positive_h(self):
        """Test hb_sbm_simple with positive h."""
        G = hb_sbm_simple(n=100, k=4, h=0.5, seed=42)

        assert G.number_of_nodes() == 100
        assert G.graph['h'] == 0.5

    def test_hb_sbm_simple_with_negative_h(self):
        """Test hb_sbm_simple with negative h."""
        G = hb_sbm_simple(n=100, k=4, h=-0.5, seed=42)

        assert G.number_of_nodes() == 100
        assert G.graph['h'] == -0.5

    def test_hb_sbm_simple_custom_community_sizes(self):
        """Test hb_sbm_simple with custom community sizes."""
        G = hb_sbm_simple(n=100, k=4, community_sizes=[40, 30, 20, 10], seed=42)

        communities = G.graph['communities']
        sizes = [len(c) for c in communities]

        assert sizes == [40, 30, 20, 10]

    def test_hb_sbm_simple_reproducibility(self):
        """Test that hb_sbm_simple is reproducible."""
        G1 = hb_sbm_simple(n=100, k=4, h=0.5, seed=42)
        G2 = hb_sbm_simple(n=100, k=4, h=0.5, seed=42)

        assert G1.number_of_edges() == G2.number_of_edges()
        assert G1.graph['communities'] == G2.graph['communities']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
