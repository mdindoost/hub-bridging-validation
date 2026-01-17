# Experiment 4: Modularity Independence Test

## Purpose

Experiment 4 validates **Theorem 4(a)** from the theoretical section:

> "For fixed degree sequence, community structure, and total inter-community edges, modularity Q is independent of hub-bridging ratio ρ_HB."

This is a **CRITICAL** experiment because it:
1. Empirically validates the mathematical proof
2. Shows hub-bridging control doesn't artificially inflate/deflate modularity
3. Demonstrates structural vs. algorithmic effects are separable

### Research Question

> When changing h (thus changing ρ_HB through rewiring), does modularity Q remain approximately constant for fixed network constraints?

### Success Criteria

- **|Pearson r| < 0.3**: Weak/no correlation between ρ_HB and Q
- **ANOVA p > 0.05** OR **effect size < 5%**: Q independent of h
- **Degree preservation**: Mean degree remains constant across h values

---

## Theoretical Background

### Why Q Should Be Independent of ρ_HB

Modularity Q depends on:
1. **|E_intra|**: Number of intra-community edges (fixed by μ parameter)
2. **Null model term**: Depends on degree sequence (fixed in HB-LFR)

Since hub-bridging rewiring:
- Keeps total inter-community edges constant
- Preserves degree sequence (in HB-LFR)
- Only changes *which specific* edges are inter-community

The modularity Q should remain unchanged!

### Mathematical Formulation

```
Q = (1/2m) Σᵢⱼ [Aᵢⱼ - kᵢkⱼ/(2m)] δ(cᵢ, cⱼ)
```

Where:
- `Aᵢⱼ` = adjacency matrix
- `kᵢ, kⱼ` = node degrees
- `m` = total edges
- `δ(cᵢ, cⱼ)` = 1 if same community, 0 otherwise

For fixed constraints, changing h only affects the distribution of inter-community edges among specific node pairs, not the total count or the null model term.

---

## Files

### Source Code

| File | Description |
|------|-------------|
| `src/validation/structural.py` | `experiment_4_modularity_independence()` function |
| `src/visualization/plots.py` | `plot_modularity_independence()` visualization |
| `src/generators/hb_lfr.py` | HB-LFR generator (preferred for exact degree preservation) |
| `src/generators/hb_sbm.py` | HB-SBM generator |
| `src/metrics/hub_bridging.py` | Hub-bridging ratio computation |

### Experiment Runners

| File | Description |
|------|-------------|
| `experiments/run_structural_validation.py` | Main runner with `--exp4` option |

### Results (in `data/results/`)

| File | Description |
|------|-------------|
| `experiment_4_results.pkl` | Raw results (pickle) |
| `figure_modularity_independence.png` | 3-panel comparison figure |
| `figure_modularity_independence.pdf` | PDF version for papers |

---

## How to Run

### Prerequisites

```bash
# Activate virtual environment
source venv/bin/activate

# Ensure dependencies installed
pip install -r requirements.txt
```

### Run Experiment 4

```bash
# Run with 30 samples per h (recommended)
python experiments/run_structural_validation.py --exp4 30

# Run with fewer samples (faster)
python experiments/run_structural_validation.py --exp4 20

# Run with more samples (more accurate)
python experiments/run_structural_validation.py --exp4 50
```

### What Gets Tested

| Generator | h Values | Samples | Networks Generated |
|-----------|----------|---------|-------------------|
| HB-LFR | [0.0, 0.5, 1.0, 1.5, 2.0] | 30 | 150 |
| **Total** | | | **150 networks** |

Note: HB-LFR is preferred because it exactly preserves degree sequences through rewiring.

---

## Experiment Parameters

### Default Settings

```python
# Parameters
generators = ['hb_lfr']          # HB-LFR preferred
h_values = [0.0, 0.5, 1.0, 1.5, 2.0]
n_samples = 30                   # Samples per h value
n = 500                          # Network size
k = 5                            # Number of communities
seed = 42                        # Base random seed
```

### HB-LFR Parameters

```python
hb_lfr(
    n=500,
    tau1=2.5,            # Degree distribution exponent
    mu=0.3,              # Fixed mixing parameter
    h=h,                 # Varied
    max_iters=5000,
    seed=sample_seed
)
```

### HB-SBM Parameters (if used)

```python
hb_sbm(
    n=500,
    k=5,
    p_in=0.3,
    p_out=0.05,
    h=h,                             # Varied
    theta_distribution='power_law',
    degree_correction_scale=1.5,
    seed=sample_seed
)
```

---

## Statistical Tests

### 1. Pearson Correlation (Primary Test)

Tests linear relationship between ρ_HB and Q:
```python
from scipy.stats import pearsonr

r, p = pearsonr(all_rho, all_Q)
# |r| < 0.2 → Strong independence
# |r| < 0.3 → Acceptable independence
# |r| > 0.3 → Dependency detected
```

### 2. Spearman Correlation (Robustness Check)

Tests monotonic relationship:
```python
from scipy.stats import spearmanr

r, p = spearmanr(all_rho, all_Q)
```

### 3. ANOVA Test (H₀: Q independent of h)

Tests whether Q varies across h values:
```python
from scipy.stats import f_oneway

F_stat, p_value = f_oneway(Q_h0, Q_h05, Q_h10, Q_h15, Q_h20)
# p > 0.05 → Cannot reject independence
# p < 0.05 → Q varies with h (check effect size)
```

### 4. Effect Size Analysis

Even if ANOVA p < 0.05, check practical significance:
```python
Q_range = max(Q_means) - min(Q_means)
relative_range = Q_range / mean(all_Q)
# relative_range < 0.05 → Practically independent
```

---

## Expected Results

### HB-LFR Expected

| Metric | Expected Value | Notes |
|--------|---------------|-------|
| Pearson r | -0.1 to +0.1 | Near-zero correlation |
| ANOVA p | > 0.05 | Q independent of h |
| Q range | < 0.02 | Minimal variation |
| Relative range | < 3% | Practically constant |

### Why Small Negative Correlation is OK

From Theorem 4(b): Higher ρ_HB slightly constrains maximum achievable Q. But for fixed constraints (Theorem 4a), correlation should be ~0.

A small negative r (e.g., -0.1) is theoretically expected and acceptable.

---

## Visualization

The experiment generates a 3-panel figure:

### Panel A: Q vs ρ_HB Scatter Plot
- All data points plotted
- Regression line with r value
- Shows correlation visually
- Flat line = independence

### Panel B: Q vs h (Error Bars)
- Mean Q at each h value
- Error bars show standard deviation
- Should be approximately flat
- Horizontal dashed line at overall mean

### Panel C: Q Coefficient of Variation
- CV for Q at each h value
- Shows consistency of Q measurements
- Low CV = reliable Q estimates

---

## Code Example

```python
from src.validation.structural import experiment_4_modularity_independence
from src.visualization.plots import plot_modularity_independence
import matplotlib.pyplot as plt

# Run experiment
results = experiment_4_modularity_independence(
    generators=['hb_lfr'],
    h_values=[0.0, 0.5, 1.0, 1.5, 2.0],
    n_samples=30,
    n=500,
    seed=42
)

# Check results
for gen in results:
    stats = results[gen]['statistics']
    assess = results[gen]['assessment']
    print(f"\n{gen.upper()}:")
    print(f"  Pearson r(ρ_HB, Q):  {stats['pearson_r']:+.4f}")
    print(f"  ANOVA p-value:       {stats['anova_p']:.4f}")
    print(f"  Q range:             {stats['Q_range']:.4f}")
    print(f"  Relative range:      {stats['relative_range']:.1%}")
    print(f"  PASS: {assess['passes_overall']}")

# Generate figure
fig = plot_modularity_independence(
    results,
    save_path='data/results/figure_modularity_independence.png'
)
plt.show()
```

---

## Loading Results

```python
import pickle

# Load results
with open('data/results/experiment_4_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Access HB-LFR results
lfr = results['hb_lfr']
stats = lfr['statistics']
print(f"Pearson r: {stats['pearson_r']:+.4f}")
print(f"ANOVA p: {stats['anova_p']:.4f}")
print(f"Q range: {stats['Q_range']:.4f}")

# Access per-h Q values
for h in lfr['h_values']:
    Q_mean = stats['Q_means'][h]
    Q_std = stats['Q_stds'][h]
    rho_mean = stats['rho_means'][h]
    print(f"h={h:.1f}: Q={Q_mean:.4f}±{Q_std:.4f}, ρ={rho_mean:.4f}")
```

---

## Interpreting Results

### PASS Criteria

Both conditions must be met:
1. **|Pearson r| < 0.3** (weak correlation)
2. **ANOVA p > 0.05** OR **effect size < 5%**

### What PASS Means

- Theorem 4(a) is empirically validated
- Hub-bridging control affects edge placement, not modularity
- Q is a valid metric independent of hub-bridging structure
- Fair to compare algorithm performance across h values

### What FAIL Means

If **|r| > 0.3**:
- Q depends on ρ_HB (contradicts Theorem 4a)
- Need to investigate generator implementation
- May indicate degree sequence is not preserved

If **ANOVA rejects with large effect**:
- Q systematically varies with h
- Check if μ parameter is truly fixed
- Verify community structure is maintained

---

## Troubleshooting

### Correlation Higher Than Expected

If |r| > 0.2:
- Check if degree sequences are preserved (should be for HB-LFR)
- Verify μ parameter is constant
- Increase sample size to reduce noise
- Try using only HB-LFR (exact degree preservation)

### ANOVA Rejects Independence

If p < 0.05 but effect size < 5%:
- This is OK - statistical vs practical significance
- Large samples can detect tiny effects
- Focus on the relative range metric

### Slow Execution

- Reduce `n_samples` (e.g., 20 instead of 30)
- HB-LFR rewiring can be slow for high h
- Each h value generates independent networks

### Different Results for HB-SBM vs HB-LFR

HB-SBM may show slightly more dependency because:
- Degree sequence is not exactly preserved
- Stochastic edge placement introduces more variation
- HB-LFR's rewiring is more controlled

---

## Relationship to Other Experiments

| Experiment | What it Tests | Related to Exp 4? |
|------------|---------------|-------------------|
| Exp 1: Parameter Control | h → ρ_HB | Yes (changes ρ_HB) |
| Exp 2: Degree Preservation | h ⊥ τ | Yes (degree constraint) |
| Exp 3: Concentration | Var(ρ_HB) | No |
| **Exp 4: Modularity** | **h ⊥ Q** | **-** |

Experiment 4 validates that Q is a valid independent metric, allowing fair comparison of community detection algorithms across different hub-bridging configurations.

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

After validating Experiment 4:

1. **Structural validation complete** (Experiments 1-4)
2. **Experiments 5-8**: Realism and algorithmic validation
3. **Publication**: Include independence results to justify fair comparisons

This completes the structural validation suite, proving that:
- h reliably controls ρ_HB (Exp 1)
- τ is independent of h (Exp 2)
- ρ_HB concentrates well (Exp 3)
- Q is independent of ρ_HB (Exp 4)
