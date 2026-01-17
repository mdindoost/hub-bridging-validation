# Hub-Bridging Network Generator Validation Framework

A comprehensive validation framework for novel hub-bridging graph generators (HB-LFR and HB-SBM) developed for PhD research.

## Overview

This framework validates that hub-bridging generators:
1. **Correctly control hub-bridging ratio** (ρ_HB) via parameter h
2. **Match real network properties** better than standard benchmarks
3. **Replicate algorithmic behaviors** observed on real networks

### What is Hub-Bridging?

Hub-bridging refers to the tendency of high-degree nodes (hubs) to form inter-community edges. The hub-bridging ratio (ρ_HB) quantifies this:

```
ρ_HB = E[d_u × d_v | inter-community edge] / E[d_u × d_v | intra-community edge]
```

- ρ_HB > 1: Hubs preferentially form bridging (inter-community) edges
- ρ_HB = 1: No hub-bridging preference
- ρ_HB < 1: Hubs preferentially form local (intra-community) edges

## Development Status

### Core Modules

| Module | Status | Description |
|--------|--------|-------------|
| **src/metrics/hub_bridging.py** | ✅ Completed | Hub-bridging ratio (ρ_HB), DSpar separation, edge classification |
| **src/metrics/network_properties.py** | ✅ Completed | Degree distribution, clustering, modularity, participation coefficient |
| **src/metrics/distance_metrics.py** | ✅ Completed | MMD, Wasserstein, KS distance for distribution comparison |
| **src/generators/hb_lfr.py** | ✅ Completed | Hub-bridging LFR generator with edge rewiring |
| **src/generators/hb_sbm.py** | ✅ Completed | Hub-bridging SBM generator |
| **src/generators/base_generators.py** | ✅ Completed | Standard LFR, SBM, Planted Partition wrappers |
| **src/generators/calibration.py** | ✅ Completed | Parameter calibration utilities |
| **src/validation/structural.py** | ✅ Completed | Experiments 1-4 (parameter control, degree preservation, modularity, concentration) |
| **src/validation/statistical_tests.py** | ✅ Completed | Monotonicity tests, multiple testing corrections, effect sizes |
| **src/algorithms/community_detection.py** | ✅ Completed | Louvain, Leiden, Label Propagation wrappers |
| **src/algorithms/sparsification.py** | ✅ Completed | DSpar (paper method), random sparsification |
| **src/visualization/plots.py** | ✅ Completed | Matplotlib plotting functions |
| **src/visualization/tables.py** | ✅ Completed | Summary table generation |

### Validation Experiments

| Experiment | Status | Description |
|------------|--------|-------------|
| **Exp 1: Parameter Control** | ✅ Completed | h → ρ_HB monotonicity with Spearman r=0.846, p<0.001 |
| **Exp 2: Degree Preservation** | ✅ Completed | Power-law exponent validation |
| **Exp 3: Modularity Independence** | ✅ Completed | Q independent of h at fixed μ |
| **Exp 4: Concentration** | ✅ Completed | Low CV for ρ_HB across samples |
| **Exp 5: Property Matching** | 🚧 In Progress | Real vs synthetic network comparison |
| **Exp 6: Network Fitting** | 🚧 In Progress | Parameter optimization for real networks |
| **Exp 7: Community Detection** | 🚧 In Progress | Algorithm performance vs h |
| **Exp 8: Sparsification** | 🚧 In Progress | DSpar behavior analysis |

#### Experiment 1 Results (HB-SBM)

| h | ρ_HB (mean ± std) | 95% CI |
|---|-------------------|--------|
| 0.00 | 1.217 ± 0.093 | ±0.033 |
| 0.50 | 1.741 ± 0.115 | ±0.041 |
| 1.00 | 2.050 ± 0.074 | ±0.026 |
| 1.50 | 2.123 ± 0.053 | ±0.019 |
| 2.00 | 2.103 ± 0.050 | ±0.018 |

- **Spearman correlation**: r = 0.846, p = 4.0e-75
- **Monotonic range**: h ∈ [0, 1.5] with expected saturation at higher h

#### Experiment 1 Results (HB-LFR)

| h | ρ_HB (mean ± std) | 95% CI |
|---|-------------------|--------|
| 0.00 | 0.834 ± 0.034 | ±0.015 |
| 0.25 | 1.247 ± 0.014 | ±0.006 |
| 0.50 | 1.463 ± 0.011 | ±0.005 |
| 0.75 | 1.653 ± 0.018 | ±0.008 |
| 1.00 | 1.810 ± 0.018 | ±0.008 |
| 1.25 | 1.931 ± 0.018 | ±0.008 |
| 1.50 | 2.033 ± 0.021 | ±0.009 |
| 1.75 | 2.120 ± 0.027 | ±0.012 |
| 2.00 | 2.194 ± 0.032 | ±0.014 |

- **Spearman correlation**: r = 0.993, p = 6.9e-170
- **Monotonic range**: h ∈ [0, 2] with no saturation
- **Rewiring-based approach**: Starts from standard LFR, rewires to target ρ_HB

### Supporting Components

| Component | Status | Description |
|-----------|--------|-------------|
| **tests/** | ✅ Completed | 41 unit tests passing |
| **config/** | ✅ Completed | YAML configuration files |
| **experiments/run_structural_validation.py** | ✅ Completed | Experiments 1-4 runner |
| **experiments/run_realism_validation.py** | 🚧 In Progress | Experiments 5-6 runner |
| **experiments/run_algorithmic_validation.py** | 🚧 In Progress | Experiments 7-8 runner |
| **notebooks/** | ⏳ Not Started | Analysis notebooks |
| **docs/** | ⏳ Not Started | Detailed documentation |

### Key Functions Implemented

```python
# Core metrics
compute_hub_bridging_ratio(G, communities)  # ρ_HB calculation
compute_dspar_separation(G, communities)     # DSpar separation metric

# Statistical tests
test_monotonicity(x_values, y_samples)       # Spearman + Jonckheere-Terpstra
bonferroni_correction(p_values, alpha)       # Multiple testing correction
fdr_correction(p_values, alpha)              # FDR (Benjamini-Hochberg)
compute_effect_size_and_ci(group1, group2)   # Cohen's d with bootstrap CI

# Generators
hb_lfr(n, mu, h,                            # Hub-bridging LFR (rewiring-based)
       max_iters=5000, tolerance=0.05)       # Rewiring parameters
hb_sbm(n, k, p_in, p_out, h,                # Hub-bridging SBM (degree-corrected)
       theta_distribution='power_law',       # 'exponential', 'power_law', 'lognormal'
       degree_correction_scale=1.5)          # Higher = more heterogeneous

# Calibration
calibrate_h_to_rho(generator, params,        # Find h → ρ_HB relationship
                   target_rho=1.5)           # Optionally find h for target ρ

# Sparsification
dspar_sparsification(G, ratio)               # Paper method (with replacement + reweighting)
```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hub-bridging-validation.git
cd hub-bridging-validation
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### Generate a Hub-Bridging SBM Network (Recommended)

```python
from src.generators.hb_sbm import hb_sbm
from src.metrics.hub_bridging import compute_hub_bridging_ratio

# Generate network with controlled hub-bridging
G, communities = hb_sbm(
    n=500,                          # Number of nodes
    k=5,                            # Number of communities
    p_in=0.3,                       # Intra-community edge probability
    p_out=0.05,                     # Inter-community edge probability
    h=1.0,                          # Hub-bridging exponent (0=standard, >0=more hub-bridging)
    theta_distribution='power_law', # Degree distribution
    degree_correction_scale=1.5,    # Degree heterogeneity
    seed=42
)

# Compute hub-bridging ratio
rho = compute_hub_bridging_ratio(G, communities)
print(f"Hub-bridging ratio: {rho:.4f}")  # Expected: ~2.0 for h=1.0
```

### Generate a Hub-Bridging LFR Network

```python
from src.generators.hb_lfr import hb_lfr
from src.metrics.hub_bridging import compute_hub_bridging_ratio

# Generate network with hub-bridging (rewiring-based)
G, communities = hb_lfr(
    n=500,            # Number of nodes
    mu=0.3,           # Mixing parameter (fraction of inter-community edges)
    h=0.5,            # Hub-bridging parameter (0=standard LFR, >0=more hub-bridging)
    max_iters=5000,   # Max rewiring iterations
    tolerance=0.05,   # Convergence tolerance for ρ_HB
    seed=42
)

# Compute hub-bridging ratio
rho = compute_hub_bridging_ratio(G, communities)
print(f"Hub-bridging ratio: {rho:.4f}")  # Expected: ~1.46 for h=0.5 (r=0.993)
```

### Run Validation Experiments

```bash
# Run HB-SBM Experiment 1 (Parameter Control Validation)
python experiments/run_structural_validation.py --improved

# Run HB-LFR Experiment 1
python experiments/run_structural_validation.py --lfr

# Compare HB-SBM vs HB-LFR generators
python experiments/run_structural_validation.py --compare

# Run structural validation (Experiments 1-4)
python experiments/run_structural_validation.py --quick

# Run full validation suite
python experiments/run_full_validation.py --quick
```

## Project Structure

```
hub-bridging-validation/
├── config/                 # Configuration files
│   ├── validation_config.yaml
│   └── network_params.yaml
├── src/                    # Source code
│   ├── metrics/           # Hub-bridging and network metrics
│   ├── generators/        # HB-LFR, HB-SBM generators
│   ├── validation/        # Validation experiments
│   ├── algorithms/        # Community detection, sparsification
│   └── visualization/     # Plotting and tables
├── experiments/           # Experiment runner scripts
├── notebooks/             # Jupyter analysis notebooks
├── tests/                 # Unit tests
├── data/                  # Data directories
│   ├── real_networks/    # Real network datasets
│   ├── generated/        # Generated synthetic networks
│   └── results/          # Validation results
└── docs/                  # Documentation
```

## Validation Experiments

### Structural Validation (Experiments 1-4)

1. **Parameter Control**: Validates that h monotonically controls ρ_HB
2. **Degree Preservation**: Verifies degree distribution is preserved
3. **Modularity Independence**: Confirms modularity is controlled by μ, not h
4. **Concentration**: Ensures ρ_HB has low variance (reliable generation)

### Realism Validation (Experiments 5-6)

5. **Property Matching**: Compares synthetic vs real network properties
6. **Network Fitting**: Optimizes parameters to match specific real networks

### Algorithmic Validation (Experiments 7-8)

7. **Community Detection**: Tests algorithm performance vs h
8. **Sparsification**: Analyzes edge sparsification behavior

## Documentation

- [Methodology](docs/methodology.md) - Detailed methodology
- [Experiments](docs/experiments.md) - Experiment specifications
- [API Reference](docs/api_reference.md) - Code documentation

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Configuration

Edit `config/validation_config.yaml` to customize:
- Sample sizes for experiments
- Statistical test parameters
- File paths for data and results

Edit `config/network_params.yaml` to customize:
- Default LFR/SBM parameters
- Parameter ranges for testing

## Citation

If you use this framework in your research, please cite:

```bibtex
@phdthesis{yourname2024hubbridging,
  title={Hub-Bridging Network Generators for Realistic Benchmark Networks},
  author={Your Name},
  year={2024},
  school={Your University}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting pull requests.

## Acknowledgments

- NetworkX development team
- python-louvain and leidenalg developers
- Your advisors and collaborators
