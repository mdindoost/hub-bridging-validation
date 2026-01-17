# Experiment 1: Parameter Control Validation

## Purpose

Experiment 1 validates that the hub-bridging parameter `h` provides **monotonic control** over the hub-bridging ratio `ρ_HB`. This is the foundational validation that proves our generators work as intended.

### Research Question

> Does increasing the hub-bridging parameter `h` reliably increase the hub-bridging ratio `ρ_HB`?

### Success Criteria

- **Spearman correlation** r > 0.8 between h and ρ_HB
- **Statistical significance** p < 0.001
- **Low variance** (tight confidence intervals)

---

## Hub-Bridging Ratio (ρ_HB)

The hub-bridging ratio quantifies whether high-degree nodes (hubs) preferentially form inter-community edges:

```
ρ_HB = E[d_u × d_v | inter-community edge] / E[d_u × d_v | intra-community edge]
```

| ρ_HB Value | Interpretation |
|------------|----------------|
| ρ_HB > 1 | Hubs preferentially bridge communities |
| ρ_HB = 1 | No hub-bridging preference |
| ρ_HB < 1 | Hubs preferentially stay local |

---

## Files

### Source Code

| File | Description |
|------|-------------|
| `src/generators/hb_sbm.py` | HB-SBM generator (degree-corrected stochastic block model) |
| `src/generators/hb_lfr.py` | HB-LFR generator (rewiring-based LFR benchmark) |
| `src/metrics/hub_bridging.py` | Hub-bridging ratio computation |
| `src/validation/structural.py` | Experiment 1 implementation |

### Experiment Runners

| File | Description |
|------|-------------|
| `experiments/run_structural_validation.py` | Main experiment runner |
| `experiments/comparative_analysis.py` | HB-SBM vs HB-LFR comparison |

### Results (in `data/results/`)

| File | Description |
|------|-------------|
| `experiment_1_hb_sbm.pkl` | HB-SBM raw results (pickle) |
| `experiment_1_hb_lfr.pkl` | HB-LFR raw results (pickle) |
| `generator_comparison.pkl` | Combined comparison data |
| `figure_generator_comparison.png` | Publication-quality comparison figure |
| `figure_generator_comparison.pdf` | PDF version for papers |
| `table_generator_comparison.tex` | LaTeX table for papers |

### Results (in `data/results/structural/`)

| File | Description |
|------|-------------|
| `exp1_parameter_control_hb_lfr_*.json` | Detailed JSON results |
| `exp1_rho_vs_h_hb_lfr.pdf` | Individual ρ vs h plot |

---

## How to Run

### Prerequisites

```bash
# Activate virtual environment
source venv/bin/activate

# Ensure dependencies installed
pip install -r requirements.txt
```

### Run Individual Generators

```bash
# Run HB-SBM Experiment 1 (improved version)
python experiments/run_structural_validation.py --improved

# Run HB-LFR Experiment 1
python experiments/run_structural_validation.py --lfr

# Run quick validation (fewer samples)
python experiments/run_structural_validation.py --quick
```

### Run Comparative Analysis

```bash
# Generate comparison figure and tables
python experiments/comparative_analysis.py
```

### Run Full Validation Suite

```bash
# Run all structural experiments (1-4)
python experiments/run_structural_validation.py

# Run with comparison
python experiments/run_structural_validation.py --compare
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--improved` | Run improved HB-SBM (recommended) |
| `--lfr` | Run HB-LFR generator |
| `--compare` | Compare HB-SBM vs HB-LFR |
| `--quick` | Quick run with fewer samples |
| `--n-samples N` | Number of samples per h value |

---

## Experiment Parameters

### Default Settings

```python
# h values tested
h_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

# Samples per h value
n_samples = 10

# Network parameters
n = 500          # Number of nodes
k = 5            # Number of communities (SBM)
mu = 0.3         # Mixing parameter (LFR)
```

### HB-SBM Parameters

```python
hb_sbm(
    n=500,
    k=5,
    p_in=0.3,                        # Intra-community edge probability
    p_out=0.05,                      # Inter-community edge probability
    h=1.0,                           # Hub-bridging parameter
    theta_distribution='power_law',  # Degree distribution
    degree_correction_scale=1.5,     # Degree heterogeneity
    seed=42
)
```

### HB-LFR Parameters

```python
hb_lfr(
    n=500,
    tau1=2.5,          # Degree distribution exponent
    tau2=1.5,          # Community size exponent
    mu=0.3,            # Mixing parameter
    h=1.0,             # Hub-bridging parameter
    max_iters=5000,    # Max rewiring iterations
    tolerance=0.05,    # Convergence tolerance
    seed=42
)
```

---

## Results Summary

### HB-SBM Results

| h | ρ_HB (mean ± std) | 95% CI |
|---|-------------------|--------|
| 0.00 | 1.194 ± 0.074 | ±0.046 |
| 0.25 | 1.447 ± 0.117 | ±0.072 |
| 0.50 | 1.713 ± 0.182 | ±0.113 |
| 0.75 | 1.868 ± 0.077 | ±0.048 |
| 1.00 | 2.021 ± 0.117 | ±0.073 |
| 1.25 | 2.118 ± 0.090 | ±0.056 |
| 1.50 | 2.134 ± 0.057 | ±0.036 |
| 1.75 | 2.139 ± 0.050 | ±0.031 |
| 2.00 | 2.123 ± 0.105 | ±0.065 |

- **Spearman r = 0.865**, p = 5.01e-28
- **Saturation observed** at h ≈ 1.5

### HB-LFR Results

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

- **Spearman r = 0.993**, p = 6.94e-170
- **No saturation** - monotonic throughout h ∈ [0, 2]

### Comparison Summary

| Metric | HB-SBM | HB-LFR | Winner |
|--------|--------|--------|--------|
| Spearman r | 0.865 | **0.993** | HB-LFR |
| Mean 95% CI | ±0.060 | **±0.009** | HB-LFR (6.7x tighter) |
| Max ρ_HB | 2.139 | **2.194** | HB-LFR |
| Control Range | 0.945 | **1.360** | HB-LFR |
| Baseline (h=0) | 1.194 | 0.834 | Different |

---

## Interpreting Results

### Why Different Baselines?

- **HB-SBM (ρ=1.194 at h=0)**: Degree correction naturally creates some hub-bridging even without explicit control
- **HB-LFR (ρ=0.834 at h=0)**: Standard LFR tends toward hub-locality (hubs connect within communities)

### Why HB-LFR Has Better Correlation?

HB-LFR uses **targeted edge rewiring** that directly optimizes for the target ρ_HB, resulting in:
- More precise control (r=0.993)
- Tighter confidence intervals
- No saturation in tested range

### Why HB-SBM Saturates?

HB-SBM modifies edge probabilities stochastically. At high h values:
- Most inter-community edges already involve hubs
- Further increases in h have diminishing returns
- Natural ceiling around ρ ≈ 2.1

---

## Code Example

```python
from src.generators.hb_lfr import hb_lfr
from src.generators.hb_sbm import hb_sbm
from src.metrics.hub_bridging import compute_hub_bridging_ratio

# Generate HB-LFR network with h=1.0
G, communities = hb_lfr(n=500, mu=0.3, h=1.0, seed=42)
rho = compute_hub_bridging_ratio(G, communities)
print(f"HB-LFR (h=1.0): ρ_HB = {rho:.3f}")  # Expected: ~1.81

# Generate HB-SBM network with h=1.0
G, communities = hb_sbm(n=500, k=5, h=1.0, seed=42)
rho = compute_hub_bridging_ratio(G, communities)
print(f"HB-SBM (h=1.0): ρ_HB = {rho:.3f}")  # Expected: ~2.02
```

---

## Loading Results

```python
import pickle

# Load HB-LFR results
with open('data/results/experiment_1_hb_lfr.pkl', 'rb') as f:
    lfr_results = pickle.load(f)

print(f"h values: {lfr_results['h_values']}")
print(f"Spearman r: {lfr_results['spearman_r']:.3f}")
print(f"ρ_HB means: {lfr_results['rho_means']}")

# Load comparison data
with open('data/results/generator_comparison.pkl', 'rb') as f:
    comparison = pickle.load(f)
```

---

## Citation

If you use these generators in your research:

```bibtex
@phdthesis{hubbridging2024,
  title={Hub-Bridging Network Generators for Realistic Benchmark Networks},
  author={Your Name},
  year={2024},
  school={Your University}
}
```

---

## Troubleshooting

### LFR Generation Fails

```
networkx.exception.ExceededMaxIterations: Could not match average_degree
```

**Solution**: Increase network size (n ≥ 250) or adjust degree constraints.

### Low Correlation

If Spearman r < 0.8:
- Increase `n_samples` for more reliable estimates
- Check network size (n ≥ 500 recommended)
- Verify community structure is present

### Slow Execution

- Use `--quick` flag for faster testing
- Reduce `n_samples` during development
- HB-LFR rewiring can be slow for large networks

---

## Next Steps

After validating Experiment 1:

1. **Experiment 2**: Degree preservation validation
2. **Experiment 3**: Modularity independence
3. **Experiment 4**: Concentration (low variance)
4. **Experiments 5-8**: Realism and algorithmic validation
