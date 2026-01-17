# Experiment 2: Degree Distribution Preservation

## Purpose

Experiment 2 validates that the hub-bridging parameter `h` does **NOT affect** the degree distribution. This proves that hub-bridging control is about **edge placement**, not degree structure.

### Research Question

> Does the power-law exponent τ remain constant across different h values?

### Success Criteria

- **ANOVA p > 0.05**: τ is independent of h (no significant difference across h values)
- **τ close to target**: Overall τ within ±0.3 of target (typically 2.5)
- **Positive KS statistics**: Power-law fits better than exponential

---

## Why This Matters

This experiment is critical because:

1. **Real networks have power-law degrees** (τ ∈ [2, 3])
2. **If h changes degree distribution** → not a fair comparison between h values
3. **We claim hub-bridging is about EDGE PLACEMENT**, not degree structure
4. **Validates orthogonality**: h controls ρ_HB independently of τ

---

## Hub-Bridging vs Degree Structure

| Property | Controlled By | Should Change with h? |
|----------|---------------|----------------------|
| Hub-bridging ratio (ρ_HB) | Parameter h | ✓ Yes (Experiment 1) |
| Power-law exponent (τ) | Generator params | ✗ No (Experiment 2) |
| Mean degree ⟨k⟩ | Network size, density | ✗ No |
| Community structure | μ parameter | ✗ No |

---

## Files

### Source Code

| File | Description |
|------|-------------|
| `src/validation/structural.py` | `experiment_2_degree_preservation_full()` function |
| `src/visualization/plots.py` | `plot_degree_preservation_comparison()` visualization |
| `src/generators/hb_sbm.py` | HB-SBM generator |
| `src/generators/hb_lfr.py` | HB-LFR generator |

### Experiment Runners

| File | Description |
|------|-------------|
| `experiments/run_structural_validation.py` | Main runner with `--exp2` option |

### Results (in `data/results/`)

| File | Description |
|------|-------------|
| `experiment_2_results.pkl` | Raw results (pickle) |
| `figure_degree_preservation.png` | 3-panel comparison figure |
| `figure_degree_preservation.pdf` | PDF version for papers |

---

## How to Run

### Prerequisites

```bash
# Activate virtual environment
source venv/bin/activate

# Ensure powerlaw package is installed
pip install powerlaw
```

### Run Experiment 2

```bash
# Run with 20 samples per h (recommended)
python experiments/run_structural_validation.py --exp2 20

# Run with fewer samples (faster, less accurate)
python experiments/run_structural_validation.py --exp2 10

# Run with more samples (slower, more accurate)
python experiments/run_structural_validation.py --exp2 30
```

### What Gets Tested

| Generator | h Values | Samples | Networks Generated |
|-----------|----------|---------|-------------------|
| HB-SBM | [0.0, 0.5, 1.0, 1.5, 2.0] | 20 | 100 |
| HB-LFR | [0.0, 0.5, 1.0, 1.5, 2.0] | 20 | 100 |
| **Total** | | | **200 networks** |

---

## Experiment Parameters

### Default Settings

```python
# Parameters
generators = ['hb_sbm', 'hb_lfr']
h_values = [0.0, 0.5, 1.0, 1.5, 2.0]
n_samples = 20          # Samples per h value
n = 500                 # Network size
target_tau1 = 2.5       # Target power-law exponent
```

### HB-SBM Parameters

```python
hb_sbm(
    n=500,
    k=5,
    p_in=0.3,
    p_out=0.05,
    h=h,                             # Varied
    theta_distribution='power_law',  # Creates power-law degrees
    degree_correction_scale=1.5,
    seed=sample_seed
)
```

### HB-LFR Parameters

```python
hb_lfr(
    n=500,
    tau1=2.5,            # Target power-law exponent
    mu=0.3,
    h=h,                 # Varied
    max_iters=3000,
    seed=sample_seed
)
```

---

## Statistical Tests

### 1. Power-Law Fitting

For each generated network:
```python
from powerlaw import Fit

fit = Fit(degrees, discrete=True)
tau_estimate = fit.alpha  # Power-law exponent
```

### 2. ANOVA Test (H₀: τ independent of h)

Tests whether τ differs significantly across h values:
```python
from scipy.stats import f_oneway

F_stat, p_value = f_oneway(tau_h0, tau_h05, tau_h10, tau_h15, tau_h20)
# p > 0.05 → τ is independent of h (PASS)
# p < 0.05 → τ depends on h (FAIL)
```

### 3. One-Sample t-test (H₀: τ = target)

Tests whether overall τ matches the target:
```python
from scipy.stats import ttest_1samp

t_stat, p_value = ttest_1samp(all_tau_values, target_tau1)
# |τ_mean - target| < 0.3 → Close to target (PASS)
```

### 4. KS Statistic (Power-law vs Exponential)

Compares goodness-of-fit:
```python
R, p = fit.distribution_compare('power_law', 'exponential')
# R > 0 → Power-law fits better
# R < 0 → Exponential fits better
```

---

## Expected Results

### HB-SBM Expected

| h | τ (expected) | Notes |
|---|--------------|-------|
| 0.00 | ~2.5 ± 0.2 | Baseline |
| 0.50 | ~2.5 ± 0.2 | Should be same |
| 1.00 | ~2.5 ± 0.2 | Should be same |
| 1.50 | ~2.5 ± 0.2 | Should be same |
| 2.00 | ~2.5 ± 0.2 | Should be same |

- **ANOVA p > 0.05**: τ independent of h
- **Overall τ ≈ 2.5**: Close to target

### HB-LFR Expected

| h | τ (expected) | Notes |
|---|--------------|-------|
| 0.00 | ~2.5 ± 0.1 | From tau1 parameter |
| 0.50 | ~2.5 ± 0.1 | Rewiring preserves degrees |
| 1.00 | ~2.5 ± 0.1 | Rewiring preserves degrees |
| 1.50 | ~2.5 ± 0.1 | Rewiring preserves degrees |
| 2.00 | ~2.5 ± 0.1 | Rewiring preserves degrees |

- **ANOVA p > 0.05**: τ independent of h
- **Overall τ ≈ 2.5**: Matches tau1 parameter
- **Tighter variance**: LFR explicitly uses tau1

---

## Visualization

The experiment generates a 3-panel figure:

### Panel A: Degree Exponent Preservation
- τ estimates vs h for both generators
- Error bars show standard deviation
- Horizontal line at target τ = 2.5
- Green band shows acceptable range (±0.3)

### Panel B: Degree Distributions (Log-Log)
- Compares h=0 vs h=max for both generators
- Should overlap (same distribution)
- Reference line shows τ=2.5 slope

### Panel C: Mean Degree Stability
- Mean degree ⟨k⟩ vs h
- Should be flat (constant mean degree)
- Verifies overall network structure preserved

---

## Code Example

```python
from src.validation.structural import experiment_2_degree_preservation_full
from src.visualization.plots import plot_degree_preservation_comparison
import matplotlib.pyplot as plt

# Run experiment
results = experiment_2_degree_preservation_full(
    generators=['hb_sbm', 'hb_lfr'],
    h_values=[0.0, 0.5, 1.0, 1.5, 2.0],
    n_samples=20,
    n=500,
    target_tau1=2.5,
    seed=42
)

# Check results
for gen in ['hb_sbm', 'hb_lfr']:
    stats = results[gen]['statistics']
    print(f"\n{gen.upper()}:")
    print(f"  Overall τ: {stats['tau_overall_mean']:.3f} ± {stats['tau_overall_std']:.3f}")
    print(f"  ANOVA p: {stats['anova_p']:.4f}")
    print(f"  Independent: {stats['independent']}")
    print(f"  Close to target: {stats['close_to_target']}")
    print(f"  PASS: {stats['passes']}")

# Generate figure
fig = plot_degree_preservation_comparison(
    results,
    save_path='data/results/figure_degree_preservation.png'
)
plt.show()
```

---

## Loading Results

```python
import pickle

# Load results
with open('data/results/experiment_2_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Access HB-LFR results
lfr_stats = results['hb_lfr']['statistics']
print(f"HB-LFR τ: {lfr_stats['tau_overall_mean']:.3f}")
print(f"ANOVA p: {lfr_stats['anova_p']:.4f}")
print(f"Passes: {lfr_stats['passes']}")

# Access per-h values
tau_means = lfr_stats['tau_means']
for h, tau in tau_means.items():
    print(f"  h={h}: τ={tau:.3f}")
```

---

## Interpreting Results

### PASS Criteria

Both conditions must be met:
1. **τ independent of h** (ANOVA p > 0.05)
2. **τ close to target** (|τ_mean - 2.5| < 0.3)

### What PASS Means

- Hub-bridging rewiring does NOT change degree distribution
- h only affects edge placement, not node degrees
- Fair comparison across h values

### What FAIL Means

If **ANOVA p < 0.05**:
- τ changes with h → degree distribution is affected
- Need to investigate generator implementation

If **τ far from target**:
- Generator not producing expected power-law
- Check generator parameters

---

## Troubleshooting

### "powerlaw package not installed"

```bash
pip install powerlaw
```

### Slow Execution

- Reduce `n_samples` (e.g., 10 instead of 20)
- HB-LFR rewiring is slower for high h values
- Consider running overnight for full validation

### High Variance in τ Estimates

- Increase network size (`n=1000`)
- Increase samples (`n_samples=30`)
- Power-law fitting is sensitive to small networks

### ANOVA Shows Dependence

If p < 0.05 (τ depends on h):
- Check if effect size is meaningful
- Small p can occur with large samples even for tiny effects
- Look at actual τ values - are differences practically significant?

---

## Relationship to Other Experiments

| Experiment | What it Tests | Depends on Exp 2? |
|------------|---------------|-------------------|
| Exp 1: Parameter Control | h → ρ_HB | No |
| **Exp 2: Degree Preservation** | **h ⊥ τ** | **-** |
| Exp 3: Modularity Independence | h ⊥ Q | No |
| Exp 4: Concentration | Low variance | No |
| Exp 5-8: Realism/Algorithms | Various | Yes (assumes τ stable) |

Experiment 2 is foundational - if τ changes with h, all comparisons across h values are confounded.

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

After validating Experiment 2:

1. **Experiment 3**: Modularity independence (Q independent of h)
2. **Experiment 4**: Concentration (low CV for ρ_HB)
3. **Experiments 5-8**: Realism and algorithmic validation
