#!/usr/bin/env python3
"""
Comprehensive diagnostic for problematic networks in Experiment 5.
Investigates why ca-CondMat, ca-AstroPh, and wiki-Vote have issues.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import networkx as nx
from collections import Counter

from src.data.network_loader import load_networks_for_experiment_5
from src.generators.hb_lfr import hb_lfr
from src.generators.calibration import extract_lfr_params_from_real
from src.metrics.hub_bridging import compute_hub_bridging_ratio

# Networks to investigate
PROBLEM_NETWORKS = ['ca-CondMat', 'ca-AstroPh', 'wiki-Vote']
WORKING_NETWORKS = ['email-Eu-core', 'ca-GrQc', 'facebook_combined']

def analyze_degree_distribution(G, name):
    """Analyze degree distribution and fit power law."""
    degrees = [d for n, d in G.degree()]

    print(f"\n  Degree Distribution:")
    print(f"    Min: {min(degrees)}")
    print(f"    Max: {max(degrees)}")
    print(f"    Mean: {np.mean(degrees):.2f}")
    print(f"    Median: {np.median(degrees):.2f}")
    print(f"    Std: {np.std(degrees):.2f}")

    # Degree percentiles
    p90 = np.percentile(degrees, 90)
    p95 = np.percentile(degrees, 95)
    p99 = np.percentile(degrees, 99)
    print(f"    90th percentile: {p90:.0f}")
    print(f"    95th percentile: {p95:.0f}")
    print(f"    99th percentile: {p99:.0f}")

    # Count hubs (nodes with degree > 2*mean)
    mean_deg = np.mean(degrees)
    n_hubs = sum(1 for d in degrees if d > 2 * mean_deg)
    n_strong_hubs = sum(1 for d in degrees if d > 5 * mean_deg)
    print(f"    Hubs (>2x mean): {n_hubs} ({100*n_hubs/len(degrees):.1f}%)")
    print(f"    Strong hubs (>5x mean): {n_strong_hubs} ({100*n_strong_hubs/len(degrees):.1f}%)")

    # Fit power law manually (simple estimation)
    try:
        from scipy import stats
        degrees_nonzero = [d for d in degrees if d > 0]
        log_degrees = np.log(degrees_nonzero)

        # Use MLE for power law exponent
        xmin = max(1, int(np.percentile(degrees_nonzero, 10)))
        degrees_above_xmin = [d for d in degrees_nonzero if d >= xmin]
        if len(degrees_above_xmin) > 10:
            alpha = 1 + len(degrees_above_xmin) / sum(np.log(d / (xmin - 0.5)) for d in degrees_above_xmin)
            print(f"    Estimated tau1 (power-law alpha): {alpha:.3f}")
    except Exception as e:
        print(f"    Power-law fit failed: {e}")

    return degrees

def analyze_community_structure(G, communities, name):
    """Analyze community structure."""
    comm_sizes = [len(c) for c in communities]

    print(f"\n  Community Structure:")
    print(f"    Num communities: {len(communities)}")
    print(f"    Min size: {min(comm_sizes)}")
    print(f"    Max size: {max(comm_sizes)}")
    print(f"    Mean size: {np.mean(comm_sizes):.2f}")
    print(f"    Median size: {np.median(comm_sizes):.2f}")
    print(f"    Std size: {np.std(comm_sizes):.2f}")

    # Size distribution
    tiny = sum(1 for s in comm_sizes if s < 5)
    small = sum(1 for s in comm_sizes if 5 <= s < 20)
    medium = sum(1 for s in comm_sizes if 20 <= s < 100)
    large = sum(1 for s in comm_sizes if s >= 100)
    print(f"    Tiny (<5): {tiny}")
    print(f"    Small (5-19): {small}")
    print(f"    Medium (20-99): {medium}")
    print(f"    Large (>=100): {large}")

    # Giant community check
    total_nodes = sum(comm_sizes)
    largest_frac = max(comm_sizes) / total_nodes
    print(f"    Largest community fraction: {100*largest_frac:.1f}%")

    if largest_frac > 0.5:
        print(f"    WARNING: Giant community detected!")

    # Compute mixing parameter (mu)
    node_to_comm = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = i

    inter_edges = 0
    total_edges = 0
    for u, v in G.edges():
        if u in node_to_comm and v in node_to_comm:
            total_edges += 1
            if node_to_comm[u] != node_to_comm[v]:
                inter_edges += 1

    mu = inter_edges / total_edges if total_edges > 0 else 0
    print(f"    Mixing parameter (mu): {mu:.4f}")

    return comm_sizes, mu

def test_generation(params, name, n_attempts=5):
    """Test LFR generation with given parameters."""
    print(f"\n  Generation Test (n_attempts={n_attempts}):")

    successes = 0
    rho_values = []
    errors = []

    for i in range(n_attempts):
        try:
            G, communities = hb_lfr(
                n=min(params.get('n', 1000), 5000),  # Cap at 5000 for speed
                tau1=params.get('tau1', 2.5),
                tau2=params.get('tau2', 1.5),
                mu=params.get('mu', 0.3),
                average_degree=params.get('average_degree', 20),
                max_degree=params.get('max_degree', 500),
                min_community=params.get('min_community', 20),
                max_community=params.get('max_community', 500),
                h=0.0,
                seed=42 + i,
                max_iters=1000,
            )
            successes += 1
            rho = compute_hub_bridging_ratio(G, communities)
            rho_values.append(rho)
        except Exception as e:
            errors.append(str(e)[:100])

    print(f"    Success rate: {successes}/{n_attempts} ({100*successes/n_attempts:.0f}%)")
    if rho_values:
        print(f"    rho_HB values: {[f'{r:.3f}' for r in rho_values]}")
        print(f"    rho_HB mean: {np.mean(rho_values):.3f}")

    if errors:
        error_counts = Counter(errors)
        print(f"    Errors:")
        for err, count in error_counts.most_common(3):
            print(f"      ({count}x) {err}")

    return successes, rho_values, errors

def test_h_range(params, name, h_values=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]):
    """Test generation across h range to find achievable rho."""
    print(f"\n  H-Range Test:")

    results = {}
    for h in h_values:
        try:
            G, communities = hb_lfr(
                n=min(params.get('n', 1000), 3000),
                tau1=params.get('tau1', 2.5),
                tau2=params.get('tau2', 1.5),
                mu=params.get('mu', 0.3),
                average_degree=params.get('average_degree', 20),
                max_degree=params.get('max_degree', 500),
                min_community=params.get('min_community', 20),
                max_community=params.get('max_community', 500),
                h=h,
                seed=42,
                max_iters=2000,
            )
            rho = compute_hub_bridging_ratio(G, communities)
            results[h] = rho
            print(f"    h={h:.1f} -> rho={rho:.3f}")
        except Exception as e:
            print(f"    h={h:.1f} -> FAILED: {str(e)[:60]}")
            results[h] = None

    valid_rhos = [r for r in results.values() if r is not None]
    if valid_rhos:
        print(f"    Achievable range: [{min(valid_rhos):.3f}, {max(valid_rhos):.3f}]")

    return results

def diagnose_network(name, G, communities):
    """Run full diagnostic on a single network."""
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC: {name}")
    print(f"{'='*70}")

    # Basic stats
    print(f"\n  Basic Statistics:")
    print(f"    Nodes: {G.number_of_nodes()}")
    print(f"    Edges: {G.number_of_edges()}")
    print(f"    Density: {nx.density(G):.6f}")
    print(f"    Avg clustering: {nx.average_clustering(G):.4f}")

    # Compute real rho_HB
    rho_real = compute_hub_bridging_ratio(G, communities)
    print(f"    Real rho_HB: {rho_real:.4f}")

    # Degree analysis
    degrees = analyze_degree_distribution(G, name)

    # Community analysis
    comm_sizes, mu = analyze_community_structure(G, communities, name)

    # Extract LFR parameters
    print(f"\n  Extracted LFR Parameters:")
    try:
        params = extract_lfr_params_from_real(G, communities)
        for k, v in sorted(params.items()):
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")
    except Exception as e:
        print(f"    FAILED: {e}")
        params = {
            'n': G.number_of_nodes(),
            'tau1': 2.5,
            'tau2': 1.5,
            'mu': mu,
            'average_degree': np.mean(degrees),
            'min_degree': max(1, int(np.percentile(degrees, 5))),
            'max_degree': int(np.percentile(degrees, 99)),
            'min_community': min(comm_sizes),
            'max_community': max(comm_sizes),
        }

    # Test generation with extracted params
    print(f"\n  === TEST 1: Generation with EXTRACTED params ===")
    test_generation(params, name)

    # Test generation with FIXED tau values
    params_fixed = params.copy()
    params_fixed['tau1'] = 2.5
    params_fixed['tau2'] = 1.5
    print(f"\n  === TEST 2: Generation with FIXED tau1=2.5, tau2=1.5 ===")
    test_generation(params_fixed, name)

    # Test generation with relaxed community sizes
    params_relaxed = params_fixed.copy()
    params_relaxed['min_community'] = 10
    params_relaxed['max_community'] = max(500, params.get('max_community', 500))
    print(f"\n  === TEST 3: Generation with RELAXED community sizes ===")
    test_generation(params_relaxed, name)

    # Test h-range with best params
    print(f"\n  === TEST 4: H-range achievability (with fixed tau) ===")
    test_h_range(params_fixed, name)

    return {
        'name': name,
        'n': G.number_of_nodes(),
        'm': G.number_of_edges(),
        'rho_real': rho_real,
        'params': params,
        'mu': mu,
        'tau1': params.get('tau1', 2.5),
        'tau2': params.get('tau2', 1.5),
        'n_communities': len(communities),
        'max_comm_frac': max(comm_sizes) / sum(comm_sizes),
    }

def main():
    print("="*70)
    print("COMPREHENSIVE NETWORK DIAGNOSTIC")
    print("="*70)

    # Load all networks
    print("\nLoading networks...")
    all_networks = PROBLEM_NETWORKS + WORKING_NETWORKS
    networks = load_networks_for_experiment_5(
        data_dir='data/real_networks',
        min_nodes=100,
        max_nodes=50000,
        network_names=all_networks,
    )

    print(f"Loaded {len(networks)} networks")

    results = {}

    # Diagnose problem networks
    print("\n" + "="*70)
    print("PROBLEM NETWORKS")
    print("="*70)
    for name in PROBLEM_NETWORKS:
        matching = [n for n in networks.keys() if name.lower() in n.lower()]
        if matching:
            net_name = matching[0]
            G = networks[net_name]['G']
            communities = networks[net_name]['communities']
            results[name] = diagnose_network(name, G, communities)
        else:
            print(f"\n{name}: NOT FOUND")

    # Diagnose working networks for comparison
    print("\n" + "="*70)
    print("WORKING NETWORKS (for comparison)")
    print("="*70)
    for name in WORKING_NETWORKS:
        matching = [n for n in networks.keys() if name.lower() in n.lower()]
        if matching:
            net_name = matching[0]
            G = networks[net_name]['G']
            communities = networks[net_name]['communities']
            results[name] = diagnose_network(name, G, communities)
        else:
            print(f"\n{name}: NOT FOUND")

    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)

    print(f"\n{'Network':<20} {'n':>8} {'m':>10} {'rho_real':>10} {'tau1':>8} {'tau2':>8} {'mu':>8} {'n_comm':>8} {'max_frac':>10}")
    print("-"*100)

    for name in PROBLEM_NETWORKS + WORKING_NETWORKS:
        if name in results:
            r = results[name]
            print(f"{name:<20} {r['n']:>8} {r['m']:>10} {r['rho_real']:>10.3f} {r['tau1']:>8.3f} {r['tau2']:>8.3f} {r['mu']:>8.3f} {r['n_communities']:>8} {r['max_comm_frac']:>10.1%}")

    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    # Identify issues
    for name in PROBLEM_NETWORKS:
        if name in results:
            r = results[name]
            issues = []

            if r['tau1'] > 3.0:
                issues.append(f"tau1={r['tau1']:.2f} is HIGH (steep degree dist, few hubs)")
            if r['tau1'] < 2.0:
                issues.append(f"tau1={r['tau1']:.2f} is LOW (flat degree dist)")
            if r['tau2'] > 2.0:
                issues.append(f"tau2={r['tau2']:.2f} is HIGH (many small communities)")
            if r['tau2'] < 1.0:
                issues.append(f"tau2={r['tau2']:.2f} is LOW")
            if r['mu'] > 0.5:
                issues.append(f"mu={r['mu']:.2f} is HIGH (weak community structure)")
            if r['mu'] < 0.1:
                issues.append(f"mu={r['mu']:.2f} is LOW (very strong communities)")
            if r['max_comm_frac'] > 0.4:
                issues.append(f"Giant community: {r['max_comm_frac']:.0%} of nodes")
            if r['n_communities'] > 100:
                issues.append(f"Many communities: {r['n_communities']}")

            print(f"\n{name}:")
            if issues:
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print(f"  - No obvious structural issues")

if __name__ == '__main__':
    main()
