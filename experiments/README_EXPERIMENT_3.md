# Experiment 3: Concentration and Reproducibility

## Purpose

Experiment 3 validates that the hub-bridging ratio `ρ_HB` **concentrates** around its expected value with **low variance**. This proves our generators produce reliable, reproducible results for controlled experiments.

### Research Question

> Given a fixed hub-bridging parameter `h`, how tightly does the measured `ρ_HB` concentrate across independent network generations?

### Success Criteria

- **CV < 15%**: Coefficient of Variation is below 15% (tight concentration)
- **Outliers < 5%**: Fewer than 5% of samples are statistical outliers
- **Normality**: Distribution approximately normal (Shapiro-Wilk p > 0.05)

---

## Why This Matters

This experiment is critical because:

1. **Reproducibility**: Science requires experiments to be reproducible
2. **Statistical power**: Low variance means smaller sample sizes needed
3. **Fair comparisons**: Tight concentration enables reliable comparisons between generators
4. **Generator quality**: High variance would indicate unstable or unreliable generation

---

## Concentration Metrics

### Coefficient of Variation (CV)

The CV measures relative variability:

```
CV = σ / μ × 100%
```

| CV Value | Interpretation |
|----------|----------------|
| CV < 5% | Excellent concentration |
| CV < 10% | Good concentration |
| CV < 15% | Acceptable concentration |
| CV > 15% | High variance (concerning) |

### Interquartile Range (IQR)

The IQR measures the spread of the middle 50% of data:

```
IQR = Q75 - Q25
```

### Outlier Detection

Using the IQR method:
- Lower fence: Q25 - 1.5 × IQR
- Upper fence: Q75 + 1.5 × IQR
- Values outside fences are outliers

---

## Files

### Source Code

| File | Description |
|------|-------------|
| `src/validation/structural.py` | `experiment_3_concentration()` function |
| `src/generators/hb_sbm.py` | HB-SBM generator |
| `src/generators/hb_lfr.py` | HB-LFR generator |
| `src/metrics/hub_bridging.py` | Hub-bridging ratio computation |

### Experiment Runners

| File | Description |
|------|-------------|
| `experiments/run_structural_validation.py` | Main runner with `--exp3` option |

### Results (in `data/results/`)

| File | Description |
|------|-------------|
| `experiment_3_results.pkl` | Raw results (pickle) |
| `figure_concentration.png` | Histogram comparison figure |
| `figure_concentration.pdf` | PDF version for papers |

---

## How to Run

### Prerequisites

```bash
# Activate virtual environment
source venv/bin/activate

# Ensure dependencies installed
pip install -r requirements.txt
```

### Run Experiment 3

```bash
# Run with 100 samples (recommended)
python experiments/run_structural_validation.py --exp3 100

# Run with fewer samples (faster, less accurate)
python experiments/run_structural_validation.py --exp3 50

# Run with more samples (slower, more accurate)
python experiments/run_structural_validation.py --exp3 200
```

### What Gets Tested

| Generator | h Value | Samples | Networks Generated |
|-----------|---------|---------|-------------------|
| HB-SBM | 1.0 | 100 | 100 |
| HB-LFR | 1.0 | 100 | 100 |
| **Total** | | | **200 networks** |

---

## Experiment Parameters

### Default Settings

```python
# Parameters
generators = ['hb_sbm', 'hb_lfr']
h_test = 1.0             # Fixed h value (mid-range)
n_samples = 100          # Independent generations
n = 500                  # Network size
k = 5                    # Number of communities
seed = 42                # Base random seed
```

### HB-SBM Parameters

```python
hb_sbm(
    n=500,
    k=5,
    p_in=0.3,
    p_out=0.05,
    h=1.0,                           # Fixed test value
    theta_distribution='power_law',
    degree_correction_scale=1.5,
    seed=sample_seed                 # Different per sample
)
```

### HB-LFR Parameters

```python
hb_lfr(
    n=500,
    tau1=2.5,
    mu=0.3,
    h=1.0,                # Fixed test value
    max_iters=5000,
    seed=sample_seed      # Different per sample
)
```

---

## Statistical Tests

### 1. Coefficient of Variation

Measures relative variability:
```python
cv = std / mean
# CV < 0.15 (15%) → PASS
```

### 2. Outlier Detection (IQR Method)

```python
iqr = q75 - q25
lower_fence = q25 - 1.5 * iqr
upper_fence = q75 + 1.5 * iqr
outliers = [x for x in samples if x < lower_fence or x > upper_fence]
# outlier_rate < 0.05 (5%) → PASS
```

### 3. Shapiro-Wilk Normality Test

Tests if distribution is normal:
```python
from scipy.stats import shapiro
W_stat, p_value = shapiro(samples)
# p > 0.05 → Approximately normal
```

### 4. Confidence Intervals

```python
ci_95 = 1.96 * std / sqrt(n_samples)
ci_99 = 2.576 * std / sqrt(n_samples)
```

---

## Expected Results

### HB-SBM Expected

| Metric | Expected Value | Notes |
|--------|---------------|-------|
| Mean ρ_HB | ~2.0 | At h=1.0 |
| CV | < 10% | Moderate variance |
| Outliers | < 5% | Few outliers |
| Shapiro p | > 0.05 | Normal distribution |

### HB-LFR Expected

| Metric | Expected Value | Notes |
|--------|---------------|-------|
| Mean ρ_HB | ~1.8 | At h=1.0 |
| CV | < 5% | **Lower** variance than SBM |
| Outliers | < 3% | Very few outliers |
| Shapiro p | > 0.05 | Normal distribution |

### Why HB-LFR Should Have Lower CV

HB-LFR uses targeted rewiring that directly optimizes for the target ρ_HB:
- More deterministic process
- Iteratively adjusts toward target
- Less stochastic variation

HB-SBM uses stochastic edge placement:
- Probabilistic edge formation
- More inherent randomness
- Higher variance expected

---

## Visualization

The experiment generates a 2-panel histogram figure:

### Panel 1: HB-SBM Distribution
- Histogram of ρ_HB values
- Vertical line at mean
- Dashed lines at ±1 standard deviation
- Title shows CV and sample size

### Panel 2: HB-LFR Distribution
- Same layout as Panel 1
- Enables visual comparison

---

## Code Example

```python
from src.validation.structural import experiment_3_concentration
import matplotlib.pyplot as plt

# Run experiment
results = experiment_3_concentration(
    generators=['hb_sbm', 'hb_lfr'],
    h_test=1.0,
    n_samples=100,
    n=500,
    seed=42
)

# Check results
for gen in ['hb_sbm', 'hb_lfr']:
    stats = results[gen]['statistics']
    assess = results[gen]['assessment']
    print(f"\n{gen.upper()}:")
    print(f"  Mean ρ_HB: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"  CV: {stats['cv']:.2%}")
    print(f"  95% CI: ±{stats['ci_95']:.4f}")
    print(f"  Outliers: {stats['n_outliers']}/{results[gen]['n_samples']}")
    print(f"  Normal: {stats['is_normal']}")
    print(f"  PASS: {assess['passes_overall']}")

# Compare precision
cv_sbm = results['hb_sbm']['statistics']['cv']
cv_lfr = results['hb_lfr']['statistics']['cv']
print(f"\nPrecision ratio: HB-LFR is {cv_sbm/cv_lfr:.1f}x more precise")
```

---

## Loading Results

```python
import pickle

# Load results
with open('data/results/experiment_3_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Access HB-LFR results
lfr = results['hb_lfr']
print(f"Mean ρ_HB: {lfr['mean']:.4f}")
print(f"CV: {lfr['cv']:.2%}")
print(f"Samples: {lfr['n_samples']}")

# Access raw samples for custom analysis
rho_samples = lfr['rho_samples']
import numpy as np
print(f"Median: {np.median(rho_samples):.4f}")
```

---

## Interpreting Results

### PASS Criteria

Both conditions must be met:
1. **CV < 15%** (Coefficient of Variation)
2. **Outliers < 5%** (Outlier rate)

### What PASS Means

- Generator produces consistent ρ_HB values
- Results are reproducible across runs
- Safe to use for controlled experiments
- Statistical analyses will have adequate power

### What FAIL Means

If **CV > 15%**:
- High variance in ρ_HB estimates
- Need larger sample sizes for reliable results
- May indicate generator instability

If **Outliers > 5%**:
- Too many extreme values
- Distribution may have heavy tails
- Consider investigating outlier causes

---

## Troubleshooting

### Slow Execution

- Reduce `n_samples` (e.g., 50 instead of 100)
- HB-LFR rewiring is slower than HB-SBM
- With 100 samples, expect ~5-10 minutes total

### High CV Values

If CV > 15%:
- Increase network size (`n=1000`)
- Check generator parameters
- May indicate h value is in transition region

### Non-Normal Distribution

If Shapiro-Wilk p < 0.05:
- Not necessarily a problem
- Check histogram for multimodality
- May indicate bimodal behavior at this h value

### Memory Issues

If running out of memory:
- Run generators separately
- Reduce `n_samples`
- Use smaller network size

---

## Relationship to Other Experiments

| Experiment | What it Tests | Uses Concentration? |
|------------|---------------|---------------------|
| Exp 1: Parameter Control | h → ρ_HB | Uses mean ρ_HB |
| Exp 2: Degree Preservation | h ⊥ τ | No |
| **Exp 3: Concentration** | **Var(ρ_HB)** | **-** |
| Exp 4: Modularity | h ⊥ Q | No |
| Exp 5-8: Realism | Various | Assumes low variance |

Experiment 3 validates that variance is low enough for Experiment 1 results to be meaningful.

---

## Performance Timing

The experiment tracks generation time for each sample:

```python
# Results include timing information
perf = results['hb_lfr']['performance']
print(f"Mean generation time: {perf['mean_time']:.3f}s")
print(f"Std generation time: {perf['std_time']:.3f}s")
print(f"Total time: {perf['total_time']:.1f}s")
```

Expected times per network (n=500):
- HB-SBM: ~0.1-0.2 seconds
- HB-LFR: ~0.5-2.0 seconds (rewiring takes longer)

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

After validating Experiment 3:

1. **Experiment 4**: Modularity independence (Q independent of h)
2. **Experiments 5-8**: Realism and algorithmic validation
3. **Publication**: Include concentration results in methods section
