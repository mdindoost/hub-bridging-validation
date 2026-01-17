# Experiment 5: Real Network Property Matching

## Purpose

Experiment 5 validates that **HB-LFR with fitted h parameter matches real network properties better than standard LFR (h=0)**.

This is a **KEY** realism validation experiment because it:
1. Demonstrates practical value of hub-bridging control
2. Shows HB-LFR produces more realistic synthetic networks
3. Validates the h-fitting methodology across diverse real networks

### Research Question

> When generating synthetic networks to match a real network, does HB-LFR with fitted h produce networks that are structurally closer to the real network than standard LFR?

### Success Criteria

- **HB-LFR wins > 75%** of networks (lower distance to real properties)
- **Mean improvement > 15%** in weighted property distance
- **p-value < 0.05** (Mann-Whitney U test)
- **Works across regimes**: Improvement in both hub-bridging and hub-isolation networks

---

## Theoretical Background

### Why HB-LFR Should Outperform Standard LFR

Real networks exhibit varying hub-bridging ratios (rho_HB):
- **Hub-bridging** (rho_HB > 1): Hubs connect communities (e.g., social networks)
- **Hub-isolation** (rho_HB < 1): Hubs are embedded in communities (e.g., collaboration networks)

Standard LFR (h=0) generates networks near rho_HB = 1 (neutral). By fitting h to match the target rho_HB, HB-LFR can capture both extremes.

### The h Parameter

```
h < 0  : Hub-isolation (rho_HB < 1) - hubs become less bridging
h = 0  : Standard LFR (rho_HB ~ 1) - neutral
h > 0  : Hub-bridging (rho_HB > 1) - hubs become more bridging
h >> 1 : Extreme hub-bridging (rho_HB >> 1)
```

### rho_HB Regimes

| Regime | rho_HB Range | Example Networks |
|--------|-------------|------------------|
| Extreme Hub-Bridging | > 4.0 | wiki-Talk (10.19) |
| Strong Hub-Bridging | 2.0 - 4.0 | facebook-combined (2.69), email-Enron (3.92) |
| Moderate Hub-Bridging | 1.0 - 2.0 | wiki-Vote (1.82) |
| Hub-Neutral | 0.8 - 1.0 | Standard LFR output |
| Hub-Isolation | < 0.8 | ca-GrQc (0.48), ca-HepTh (0.61) |

### Priority Properties with Weights

The weighted distance metric prioritizes key properties:

| Property | Weight | Rationale |
|----------|--------|-----------|
| rho_HB | 3.0 | Primary target - hub-bridging is the focus |
| delta_DSpar | 2.0 | Sparsification sensitivity - key distinction |
| modularity | 1.5 | Community structure quality |
| degree_assortativity | 1.5 | Degree correlation pattern |
| clustering_avg | 1.0 | Local structure |
| power_law_alpha | 1.0 | Degree distribution shape |
| transitivity | 1.0 | Global clustering |
| avg_path_length | 0.5 | May vary with size |
| rich_club_10 | 0.5 | Top-degree connectivity |

---

## Files

### Source Code

| File | Description |
|------|-------------|
| `src/validation/realism.py` | `experiment_5_real_network_matching()`, `experiment_5_extended()` |
| `src/generators/calibration.py` | `fit_h_to_real_network()`, `fit_h_to_real_network_extended()` |
| `src/data/network_loader.py` | `load_real_networks_from_snap()`, `load_networks_for_experiment_5()` |
| `src/metrics/hub_bridging.py` | Hub-bridging ratio computation |
| `src/visualization/plots.py` | `plot_experiment_5_results()` |

### Experiment Runners

| File | Description |
|------|-------------|
| `experiments/run_realism_validation.py` | Main runner with `--exp5` option |
| `scripts/download_snap_networks.py` | Download SNAP networks |

### Results (in `data/results/realism/`)

| File | Description |
|------|-------------|
| `exp5_real_network_matching_*.pkl` | Raw results (pickle) |
| `exp5_real_network_matching_*.json` | Human-readable results |
| `figure_exp5_real_network_matching.png` | Multi-panel comparison figure |

---

## How to Run

### Prerequisites

```bash
# Activate virtual environment
source venv/bin/activate

# Ensure dependencies installed
pip install -r requirements.txt
```

### Step 1: Download Networks

```bash
# Download all 28 networks (Tier 1 + Tier 2)
python scripts/download_snap_networks.py

# Or download only essential networks (faster)
python scripts/download_snap_networks.py --quick
```

### Step 2: Run Experiment 5

```bash
# Standard mode (basic property matching)
python experiments/run_realism_validation.py --exp5

# Extended mode (weighted properties, regime analysis)
python experiments/run_realism_validation.py --exp5 --extended

# Quick test run
python experiments/run_realism_validation.py --exp5 --quick

# Specify data directory
python experiments/run_realism_validation.py --exp5 --data-dir /path/to/networks

# Use sample networks (karate club) for testing
python experiments/run_realism_validation.py --exp5 --use-sample

# Download and run in one command
python experiments/run_realism_validation.py --exp5 --download --extended
```

### Run for Specific Networks

```bash
# Recommended: 5 networks, 10 samples each
python experiments/run_realism_validation.py --exp5 --n-samples 10

# Then manually filter networks in data-dir to:
# facebook-combined, wiki-Vote, ca-GrQc, email-Enron, com-Youtube
```

### What Gets Tested

| Mode | Networks | Samples/Network | Total Generations |
|------|----------|-----------------|-------------------|
| Quick | ~5 | 5 | ~25 |
| Standard | ~17 | 30 | ~510 |
| Extended | ~28 | 30 | ~840 |

---

## Experiment Parameters

### Standard Mode Settings

```python
# For each real network:
n_synthetic_per_real = 30       # Synthetic samples per network
n_calibration_samples = 10      # Samples for h calibration
n_h_points = 25                 # Points in h grid search

# h range: [0.0, 2.0] (standard)
```

### Extended Mode Settings

```python
# Extended h range for diverse rho_HB targets
h_range = (-0.5, 3.5)           # Supports hub-isolation and extreme hub-bridging
n_h_points = 31                 # Finer grid

# Adaptive range adjustment for extreme targets
# If target rho_HB > 5: h_range extends up to 5.0
# If target rho_HB < 0.5: h_range extends down to -1.0

# Weighted distance computation
use_weighted_distance = True    # Apply property weights
```

### Quick Mode Settings

```python
n_synthetic_per_real = 5
n_calibration_samples = 5
n_h_points = 15
max_nodes = 1000                # Skip large networks
```

---

## Network Domains and Expected rho_HB

### Tier 1: Core Networks (17)

| Network | Domain | Nodes | Expected rho_HB |
|---------|--------|-------|-----------------|
| facebook-combined | Social | 4,039 | 2.69 |
| wiki-Vote | Social | 7,115 | 1.82 |
| wiki-Talk | Social | 2.4M | 10.19 |
| email-Enron | Communication | 36,692 | 3.92 |
| email-Eu-core | Communication | 1,005 | 2.15 |
| ca-GrQc | Collaboration | 5,242 | 0.48 |
| ca-HepTh | Collaboration | 9,877 | 0.61 |
| ca-HepPh | Collaboration | 12,008 | 0.55 |
| ca-CondMat | Collaboration | 23,133 | 0.59 |
| ca-AstroPh | Collaboration | 18,772 | 0.72 |
| cit-HepTh | Citation | 27,770 | 1.35 |
| cit-HepPh | Citation | 34,546 | 1.42 |
| com-Youtube | Social | 1.1M | 2.45 |
| com-DBLP | Collaboration | 317K | 0.89 |
| com-Amazon | Commerce | 334K | 1.12 |
| com-LiveJournal | Social | 4M | 2.83 |
| soc-Epinions1 | Social | 75,879 | 2.31 |

### Tier 2: Additional Networks (11)

| Network | Domain | Expected rho_HB |
|---------|--------|-----------------|
| loc-brightkite | Location | 2.1 |
| loc-gowalla | Location | 1.8 |
| soc-Slashdot | Social | 1.95 |
| web-Stanford | Web | 1.5 |
| web-NotreDame | Web | 1.3 |
| roadNet-PA | Infrastructure | 0.7 |
| roadNet-TX | Infrastructure | 0.65 |
| bio-yeast | Biological | 1.1 |
| p2p-Gnutella | P2P | 0.85 |
| as-caida | Internet | 1.4 |
| twitter-combined | Social | 3.2 |

---

## Statistical Tests

### 1. Mann-Whitney U Test (Primary)

Tests whether HB-LFR distances are significantly lower than LFR distances:

```python
from scipy.stats import mannwhitneyu

U_stat, p_value = mannwhitneyu(
    hb_distances, lfr_distances,
    alternative='less'  # HB-LFR should have LOWER distances
)
# p < 0.05 → HB-LFR significantly better
```

### 2. Win Rate Analysis

```python
n_networks = len(results)
hb_wins = sum(1 for r in results.values() if r['hb_distance'] < r['lfr_distance'])
win_rate = hb_wins / n_networks
# Target: win_rate > 0.75
```

### 3. Effect Size (Rank-Biserial Correlation)

```python
n1 = len(hb_distances)
n2 = len(lfr_distances)
effect_size = 1 - (2 * U_stat) / (n1 * n2)
# effect_size > 0.3 → meaningful improvement
```

### 4. Improvement Percentage

```python
improvements = [(lfr - hb) / lfr * 100 for hb, lfr in zip(hb_distances, lfr_distances)]
mean_improvement = np.mean(improvements)
# Target: mean_improvement > 15%
```

---

## Expected Results

### Overall Metrics

| Metric | Expected Value | Notes |
|--------|---------------|-------|
| Win rate | > 80% | HB-LFR wins for most networks |
| Mean improvement | > 20% | Substantial distance reduction |
| Mann-Whitney p | < 0.01 | Highly significant |
| Effect size | > 0.4 | Large effect |

### By Regime

| Regime | Expected Improvement | Why |
|--------|---------------------|-----|
| Extreme Hub-Bridging | 30-50% | Standard LFR far from target |
| Strong Hub-Bridging | 20-35% | Clear benefit |
| Moderate Hub-Bridging | 10-20% | Some improvement |
| Hub-Neutral | 5-10% | Less differentiation |
| Hub-Isolation | 25-40% | h < 0 needed |

### Collaboration Networks (Hub-Isolation)

Networks like ca-GrQc (rho_HB = 0.48) require **negative h** to achieve hub-isolation. This is only possible with the extended h range.

---

## Visualization

The experiment generates a multi-panel figure:

### Panel A: Distance Comparison Bar Chart
- Per-network comparison of HB-LFR vs LFR distances
- Grouped by rho_HB regime
- Shows improvement for each network

### Panel B: h Fitted Values
- Fitted h value for each network
- Colored by regime
- Shows range of h needed

### Panel C: Improvement Distribution
- Histogram of improvement percentages
- Vertical line at 0% (no improvement)
- Shows overall distribution

### Panel D: Regime Summary
- Win rate by regime
- Mean improvement by regime
- Number of networks per regime

---

## Code Example

```python
from src.validation.realism import (
    experiment_5_extended,
    summarize_experiment_5_extended,
    PRIORITY_PROPERTIES,
    RHO_REGIMES,
)
from src.data import load_networks_for_experiment_5
from src.visualization import plot_experiment_5_results
import matplotlib.pyplot as plt

# Load networks
networks = load_networks_for_experiment_5(
    data_dir='data/real_networks',
    min_nodes=100,
    max_nodes=50000,
)

# Run extended experiment
results = experiment_5_extended(
    real_networks_dict=networks,
    n_synthetic_per_real=30,
    use_extended_h_fitting=True,
    use_weighted_distance=True,
    n_calibration_samples=10,
    n_h_points=25,
    seed=42,
)

# Get summary
summary = summarize_experiment_5_extended(results)

# Print results
print(f"Networks tested: {summary['n_networks']}")
print(f"HB-LFR wins: {summary['hb_wins']}")
print(f"Mean improvement: {summary['avg_improvement_percent']:.1f}%")
print(f"Mann-Whitney p: {summary['statistical_test']['p_value']:.4f}")
print(f"Validation: {'PASS' if summary['passes'] else 'FAIL'}")

# Print by regime
for regime, data in summary['by_regime'].items():
    n = data['n_networks']
    wins = data['hb_wins']
    imp = data['mean_improvement'] * 100
    print(f"  {regime}: {wins}/{n} wins, {imp:+.1f}% improvement")

# Generate figure
fig = plot_experiment_5_results(
    results,
    save_path='data/results/realism/figure_exp5.png'
)
plt.show()
```

---

## Loading Results

```python
import pickle

# Load results
with open('data/results/realism/exp5_real_network_matching_extended_*.pkl', 'rb') as f:
    results = pickle.load(f)

# Access per-network results
for net_name, net_data in results['networks'].items():
    h = net_data['h_fitted']
    hb_dist = net_data['overall_distance_hb']
    lfr_dist = net_data['overall_distance_std']
    regime = net_data['regime']
    improvement = (lfr_dist - hb_dist) / lfr_dist * 100

    print(f"{net_name}: h={h:.2f}, regime={regime}, improvement={improvement:+.1f}%")

# Access summary
summary = results['summary']
print(f"\nOverall: {summary['hb_wins']}/{summary['n_networks']} wins")
print(f"p-value: {summary['statistical_test']['p_value']:.4f}")
```

---

## Interpreting Results

### PASS Criteria

All conditions should be met:
1. **HB-LFR wins > 75%** of networks
2. **Mean improvement > 15%**
3. **p-value < 0.05** (statistically significant)

### What PASS Means

- HB-LFR with fitted h produces more realistic synthetic networks
- Hub-bridging control is practically valuable
- The h-fitting methodology works across diverse networks
- HB-LFR is recommended for generating benchmark networks

### What FAIL Means

If **win rate < 75%**:
- Check if networks are too small (< 100 nodes)
- Verify community detection is working
- Some networks may not be community-structured

If **mean improvement < 15%**:
- May need extended h range for extreme regimes
- Check if rho_HB is achievable (see achievability reports)

If **p-value > 0.05**:
- Increase sample size
- May need more networks in dataset

---

## Troubleshooting

### "Rewiring stalled" Warnings

```
Rewiring stalled at iteration X: rho=Y.YYY
```

This is **expected behavior** when:
- Target rho_HB is very high (> 5)
- Network structure limits achievable rho_HB
- The algorithm has reached the ceiling

The experiment handles this by reporting "achievability" status.

### NaN Results

Causes:
- Network too small (< 50 nodes)
- Community detection failed
- All LFR generations failed

Solutions:
- Use `--min-nodes 100` filter
- Check network file format
- Use `--use-sample` to test with karate club

### Slow Execution

- Use `--quick` for testing
- Reduce `--n-samples` (e.g., 10 instead of 30)
- Filter to smaller networks (< 10,000 nodes)
- High h values require more rewiring iterations

### Community Detection Issues

The loader uses multiple fallback methods:
1. Louvain (NetworkX `louvain_communities`)
2. python-louvain (`community.best_partition`)
3. Leiden (if leidenalg installed)
4. Label Propagation (fallback)

Install python-louvain for best results:
```bash
pip install python-louvain
```

### Hub-Isolation Networks (h < 0)

For networks with rho_HB < 0.8 (e.g., ca-GrQc):
- Use `--extended` mode for h range (-0.5, 3.5)
- Standard mode (h >= 0) cannot achieve hub-isolation

---

## Relationship to Other Experiments

| Experiment | What it Tests | Related to Exp 5? |
|------------|---------------|-------------------|
| Exp 1: Parameter Control | h -> rho_HB | Yes (uses h fitting) |
| Exp 2: Degree Preservation | h perpendicular to tau | Yes (tau extraction) |
| Exp 3: Concentration | Var(rho_HB) | No |
| Exp 4: Modularity | h perpendicular to Q | Yes (Q is a property) |
| **Exp 5: Real Network Matching** | **HB-LFR vs LFR** | **-** |
| Exp 6: Network Fitting | Multi-parameter optimization | Builds on Exp 5 |

Experiment 5 demonstrates that the controlled h parameter (validated in Exp 1-4) provides practical value for generating realistic networks.

---

## Citation

If you use these results in your research:

```bibtex
@phdthesis{hubbridging2024,
  title={Hub-Bridging Network Generators for Realistic Benchmark Networks},
  author={Your Name},
  year={2024},
  school={Your University}
}
```

---

## Next Steps

After validating Experiment 5:

1. **Experiment 6: Network Fitting** - Multi-parameter optimization to match real networks
2. **Experiments 7-8: Algorithmic Validation** - Community detection and sparsification
3. **Publication**: Use results to justify HB-LFR for benchmark generation

This experiment establishes the **practical value** of hub-bridging control, showing that HB-LFR generates more realistic synthetic networks than standard LFR across diverse real-world networks.
