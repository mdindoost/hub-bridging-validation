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

## Key Implementation Fixes

Several critical fixes were made to ensure correct experiment behavior:

### 1. Distance Metric Fix (Unbounded Relative Errors)

**Problem:** The original distance formula used unbounded relative errors:
```python
error = abs(synth - real) / abs(real)  # Could be 30,000%+ if real ≈ 0!
```

When `delta_DSpar_real ≈ 0`, this produced errors like 31,550%, completely drowning out
the ρ_HB improvement (which was the whole point of the experiment).

**Fix:** Two-part solution in `compute_weighted_distance()`:
1. **Special handling** for bounded/near-zero properties (delta_DSpar, clustering, etc.)
2. **Capped relative errors** at 200% max for unbounded properties

### 2. Essential Parameters Only

**Problem:** Over-constraining LFR parameters (average_degree, min_community, max_community)
severely limited the achievable ρ range to [0.83, 0.91].

**Fix:** `extract_lfr_params_from_real()` now returns only essential parameters:
- n, tau1, tau2, mu (shape matters, not exact values)
- Achievable ρ range expanded to [0.56, 2.41] — a 21x improvement!

### 3. Smart 3-Phase Calibration

**Problem:** Grid search calibration was slow and wasteful.

**Fix:** Binary search with early termination:
- Phase 1: Probe achievable range (3 samples)
- Phase 2: Check if target is achievable (skip if not)
- Phase 3: Binary search only when needed
- Result: ~10x faster calibration

### 4. Parallel Execution Support

**Problem:** Sequential processing of 13+ networks took hours.

**Fix:** Added `run_exp5_parallel.py`:
- Processes multiple networks simultaneously
- Per-network log files for debugging
- Per-network CSV files (merged at end)
- 4 workers → ~4x speedup

### 5. Robust Parameter Extraction

**Problem:** Some real networks have extreme parameter values that cause LFR generation to fail:
- ca-CondMat: tau2=2.71 → LFR fails (can't generate valid community sizes)
- wiki-Vote: tau1=2.93 → Very few hubs, narrow achievable ρ range
- Result: 20% success rate for ca-CondMat, achievable ρ capped at ~0.88

**Fix:** Added `extract_lfr_params_robust()` with adaptive bounds:

```python
# Adaptive parameter bounds with minimal intervention
tau1 = min(tau1_raw, 2.8)   # Cap at 2.8 (ensures hub availability)
tau2 = min(tau2_raw, 2.0)   # Cap at 2.0 (LFR stability requirement)

# Adaptive mu boost for high ρ targets
if target_rho > 1.5:
    mu_min = 0.20 + 0.05 * (target_rho - 1.5)  # Proportional boost
    mu_min = min(mu_min, 0.35)  # Cap to preserve community structure
    mu = max(mu_raw, mu_min)
```

**Principles:**
1. **Preserve when possible** - Use raw params if they work
2. **Bound when necessary** - Apply minimal changes for feasibility
3. **Log all changes** - Full transparency for paper
4. **Justify scientifically** - Each bound has empirical reason

**Results after fix:**

| Network | Before Fix | After Fix |
|---------|------------|-----------|
| ca-CondMat success rate | 20% | **100%** |
| ca-CondMat achievable ρ | 0.88 | **2.39** (target: 2.48) |
| wiki-Vote tau1 | 2.93 (fails) | 2.80 (works) |

**New CSV columns for transparency:**
- `tau1_raw`, `tau1_used` - Original vs adjusted tau1
- `tau2_raw`, `tau2_used` - Original vs adjusted tau2
- `mu_raw`, `mu_used` - Original vs adjusted mu
- `param_adjustments` - Summary of what was changed (e.g., "tau2; mu")

**Fallback mechanism:** If adjusted params still fail, falls back to canonical
parameters (tau1=2.5, tau2=1.5) with full logging.

### 6. Achievable Range Estimation Fix

**Problem:** When the upper bound probe (h=3.5) failed during Phase 1, the code set an arbitrary placeholder:
```python
if rho_upper is None:
    rho_upper = 10.0  # Arbitrary! Led to wrong achievability decisions
```

This caused:
- Target incorrectly declared "ACHIEVABLE" when it wasn't
- Unnecessary binary search iterations (wasted ~30 minutes per network)
- Wrong achievable range reported: `[0.31, 10.0]` instead of actual `[0.31, ~1.2]`

**Fix:** Estimate `rho_upper` conservatively based on h=0 result and mixing parameter:

```python
# In calibration.py - fit_h_to_real_network_extended()
if rho_upper is None:  # Upper probe failed
    mu = lfr_params.get('mu', 0.3)
    if rho_middle is not None:
        # Factor depends on mixing parameter
        if mu < 0.2:
            factor = 1.3  # Low mixing → lower ceiling
        elif mu < 0.4:
            factor = 1.5  # Medium mixing
        else:
            factor = 2.0  # High mixing allows more rewiring headroom
        rho_upper = rho_middle * factor
```

**Phase 1 probe order changed:** Now tests h=0 (middle) FIRST, then h_max, so we have the h=0 result available for estimation.

**Results after fix (ca-HepTh example):**

| Metric | Before | After |
|--------|--------|-------|
| Achievable range | [0.31, 10.0] (wrong) | [0.78, 1.43] (correct) |
| Target decision | "ACHIEVABLE" (wrong) | "TOO HIGH" (correct) |
| Binary search | 8+ iterations (~30 min) | SKIPPED |
| Calibration time | ~63 min | ~21 min |

### 7. Target Mismatch Fix (target_rho Parameter)

**Problem:** The `hb_lfr()` function computed its own target from h using a hardcoded formula, ignoring the real network's target:

```python
# OLD: hb_lfr computed formula-based target
rho_target = 1.0 + 1.5 * (1.0 - np.exp(-0.8 * h))
# At h=1.5: formula gives target=2.048, but real target was 1.484!
```

This caused:
- Rewiring optimized for wrong target (2.048 vs 1.484)
- Early termination triggered for wrong reasons
- Confusing logs ("target=2.048" when calibration wanted 1.484)

**Fix:** Added `target_rho` parameter to `hb_lfr()`:

```python
def hb_lfr(..., target_rho: Optional[float] = None):
    if target_rho is not None:
        rho_target_value = target_rho
        logger.info(f"Target ρ_HB = {rho_target_value:.3f} (from calibration)")
    else:
        # Fall back to formula for standalone use
        rho_target_value = 1.0 + 1.5 * (1.0 - np.exp(-0.8 * h))
        logger.info(f"Target ρ_HB = {rho_target_value:.3f} (from formula, h={h:.2f})")
```

All calibration calls now pass `target_rho`:
```python
G, communities = hb_lfr(..., h=h_test, target_rho=rho_target)
```

**Results after fix:**
- Logs now show correct target: `Target ρ_HB = 1.484 (from calibration)`
- Rewiring optimizes for actual target
- Upper probe now succeeds more often (targets achievable ρ, not formula ρ)
- Backward compatible: standalone `hb_lfr()` calls still use formula

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
| degree_assortativity | 1.0 | Degree correlation pattern |
| clustering_avg | 1.0 | Local structure |
| power_law_alpha | 1.0 | Degree distribution shape |
| transitivity | 0.5 | Global clustering (lower weight - affected by rewiring) |
| avg_path_length | 0.5 | May vary with size |
| rich_club_10 | 0.5 | Top-degree connectivity |

### Weighted Distance Calculation (Fixed)

The distance metric uses **robust normalization** to prevent any single property from dominating:

```python
# Special properties: Use SCALED ABSOLUTE ERROR (not relative error)
# These properties have bounded ranges or can be near-zero
special_properties = {
    'degree_assortativity': 2.0,   # Range [-1, 1], scale factor = 2
    'modularity': 1.0,              # Range [0, 1]
    'transitivity': 1.0,            # Range [0, 1]
    'clustering_avg': 1.0,          # Range [0, 1]
    'delta_DSpar': 1.0,             # Can be near 0, range ~[-0.5, 0.5]
}

# For special properties:
error = abs(synth_val - real_val) / scale

# For other properties (rho_HB, power_law_alpha, avg_path_length, rich_club_10):
# Use CAPPED relative error (max 200%)
error = min(abs(synth_val - real_val) / abs(real_val), 2.0)

# Final weighted distance:
distance = sqrt(sum(weight * error²) / sum(weights))
```

**Why this matters:** Without capping, a property like `delta_DSpar = 0.0001` (near-zero)
could produce 30,000% relative error and completely dominate the distance, even when
ρ_HB (the most important metric) improves dramatically.

**Example - facebook_combined:**
| Metric | Old (broken) | New (fixed) |
|--------|--------------|-------------|
| HB Distance | 137.34 | 0.169 |
| Std Distance | 68.48 | 0.337 |
| Improvement | -100% (WRONG) | **+50%** (CORRECT) |

---

## Files

### Source Code

| File | Description |
|------|-------------|
| `src/validation/realism.py` | `experiment_5_real_network_matching()`, `experiment_5_extended()`, CSV export with param adjustments |
| `src/generators/calibration.py` | `fit_h_to_real_network()`, `fit_h_to_real_network_extended()`, `extract_lfr_params_robust()`, `extract_lfr_params_with_fallback()` |
| `src/data/network_loader.py` | `load_real_networks_from_snap()`, `load_networks_for_experiment_5()` |
| `src/metrics/hub_bridging.py` | Hub-bridging ratio computation |
| `src/visualization/plots.py` | `plot_experiment_5_results()` |

### Experiment Runners

| File | Description |
|------|-------------|
| `experiments/run_realism_validation.py` | Main runner with `--exp5` option (single network or sequential) |
| `experiments/run_exp5_parallel.py` | **Parallel runner** - processes multiple networks simultaneously |
| `scripts/download_snap_networks.py` | Download SNAP networks |

### Results (in `data/results/realism/`)

| File | Description |
|------|-------------|
| `exp5_extended_<timestamp>_<network>.csv` | Per-network CSV results (parallel mode) |
| `exp5_extended_COMBINED_<timestamp>.csv` | Merged CSV with all networks |
| `exp5_extended_<timestamp>.pkl` | Pickle files with full data |
| `logs/<network>.log` | Per-network execution logs (parallel mode) |
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

# Override max nodes filter (for large networks like com-amazon with 335K nodes)
python experiments/run_realism_validation.py --exp5 --extended --networks com-amazon --max-nodes 400000

# Use sample networks (karate club) for testing
python experiments/run_realism_validation.py --exp5 --use-sample

# Download and run in one command
python experiments/run_realism_validation.py --exp5 --download --extended
```

### Run in Parallel (Recommended)

For faster execution, run all networks in parallel using the parallel runner:

```bash
# Run all networks with 4 parallel workers (default: 30 samples each)
python experiments/run_exp5_parallel.py --workers 4

# With more workers (adjust based on CPU cores)
python experiments/run_exp5_parallel.py --workers 8

# Quick mode for testing (5 samples)
python experiments/run_exp5_parallel.py --workers 4 --quick

# Custom sample count
python experiments/run_exp5_parallel.py --workers 4 --n-samples 10

# Specific networks only
python experiments/run_exp5_parallel.py --workers 4 --networks email-Eu-core facebook_combined
```

**How parallel execution works:**
1. Loads all networks and sorts by size (smallest first)
2. Launches N worker processes
3. Each worker runs one network via `run_realism_validation.py --networks <name>`
4. Each network writes to its own timestamped CSV file
5. At the end, merges all CSVs into `exp5_extended_COMBINED_<timestamp>.csv`

**Output files:**
- `exp5_extended_<timestamp>_<network>.csv` - Per-network results (one file per network)
- `exp5_extended_COMBINED_<timestamp>.csv` - Merged results from all networks
- `exp5_extended_<timestamp>.pkl` - Pickle files with full data
- `logs/<network>.log` - Detailed log for each network

**Monitoring parallel execution:**
```bash
# Watch overall progress (main terminal output)
# The parallel runner shows: "✓ Completed: <network> (XXXs)"

# View a specific network's log in real-time
tail -f data/results/realism/logs/facebook_combined.log

# Check all completed networks
ls -la data/results/realism/exp5_extended_*.csv | wc -l

# View CSV results as they come in
cat data/results/realism/exp5_extended_*_facebook_combined.csv
```

**Estimated times (with 4 workers):**
| Networks | Sequential | Parallel (4 workers) |
|----------|------------|---------------------|
| 13 networks | ~3-4 hours | ~1 hour |
| Quick mode | ~30 min | ~10 min |

### Run for Specific Networks

```bash
# Recommended: 5 networks, 10 samples each
python experiments/run_realism_validation.py --exp5 --n-samples 10

# Then manually filter networks in data-dir to:
# facebook-combined, wiki-Vote, ca-GrQc, email-Enron, com-Youtube
```

### What Gets Tested

| Mode | Networks | Calibration | Validation | Est. Time |
|------|----------|-------------|------------|-----------|
| Quick | ~5 | ~18 samples (binary search) | 5 samples | ~15 min |
| Standard | ~17 | ~18 samples (binary search) | 30 samples | ~1.5 hours |
| Extended | ~11 (≤100k nodes) | ~18 samples (binary search) | 50 samples | ~1 hour |

**Note:** Binary search calibration is ~10x faster than grid search.

---

## Experiment Parameters

### Smart 3-Phase Calibration (Default)

The experiment uses a **smart 3-phase calibration** to efficiently find the optimal h parameter:

```python
# PHASE 1: Probe achievable range (3 samples)
#   - Test h_min (e.g., -0.5) → get rho_lower
#   - Test h=0 (middle) FIRST → get rho_middle (needed for estimation)
#   - Test h_max (e.g., 3.5) → get rho_upper
#   - If h_max fails: estimate rho_upper from rho_middle * factor
#   Result: Achievable range [rho_lower, rho_upper]
#
# All probes use target_rho parameter to pass real network's target
# to hb_lfr, ensuring consistent rewiring behavior

# PHASE 2: Check target achievability
#   - If target > rho_upper: TARGET TOO HIGH → skip binary search
#   - If target < rho_lower: TARGET TOO LOW → skip binary search
#   - If unreachable: Use FAST MODE (max_iters=500) for validation

# PHASE 3: Binary search (only if target is achievable)
#   - Test midpoint with 2 samples for stability
#   - If rho < target: search higher half
#   - If rho > target: search lower half
#   - Stop when range < 0.15 or diff < 0.05
# Total: ~6-18 samples per network (vs 100+ for grid search)
```

**Key Features:**
- **Smart range detection**: 3-point probe establishes achievable ρ range upfront
- **Robust estimation**: If upper probe fails, estimates from h=0 result (see Fix 6)
- **Correct target passing**: All probes pass real target to hb_lfr (see Fix 7)
- **Early skip**: Binary search skipped entirely when target is unreachable
- **FAST MODE**: Reduced iterations (500 vs 5000) when target not achievable
- **~3-10x faster** than before (skipping unnecessary iterations)

### Essential Parameters Only (with Robust Extraction)

LFR parameter extraction now returns **only essential parameters** with **adaptive bounds**:

```python
# extract_lfr_params_robust() returns:
params = {
    'n': n,           # Network size
    'tau1': tau1,     # Degree exponent (capped at 2.8 for hub availability)
    'tau2': tau2,     # Community exponent (capped at 2.0 for LFR stability)
    'mu': mu,         # Mixing parameter (boosted for high ρ targets)
    # Raw values also stored: tau1_raw, tau2_raw, mu_raw
    # Adjustments logged: adjustments dict
}
# We intentionally EXCLUDE: average_degree, min_community, max_community
```

**Why essential only?** Over-constraining parameters (like exact average_degree) severely limits
the rewiring algorithm's ability to achieve target ρ values:

**Why adaptive bounds?** Some real networks have extreme parameters that cause LFR to fail:
- tau2 > 2.5 → LFR can't generate valid community size distribution
- tau1 > 2.9 → Too few hubs, very narrow achievable ρ range

| Configuration | Achievable ρ Range | Span |
|--------------|-------------------|------|
| Over-constrained (old) | [0.83, 0.91] | 0.08 |
| **Essential only (new)** | **[0.56, 2.41]** | **1.85** |

This **21x wider range** means most real network targets are now achievable!

### Standard Mode Settings

```python
# For each real network:
n_synthetic_per_real = 30       # Synthetic samples per network
# Calibration uses binary search (~18 samples automatically)

# h range: [0.0, 2.0] (standard)
```

### Extended Mode Settings

```python
# Extended h range for diverse rho_HB targets
h_range = (-0.5, 3.5)           # Supports hub-isolation and extreme hub-bridging

# Adaptive range adjustment for extreme targets
# If target rho_HB > 5: h_range extends up to 5.0
# If target rho_HB < 0.5: h_range extends down to -1.0

# Weighted distance computation
use_weighted_distance = True    # Apply property weights
```

### Quick Mode Settings

```python
n_synthetic_per_real = 5
max_nodes = 1000                # Skip large networks
```

### Expected Calibration Output

**When target IS achievable:**
```
============================================================
CALIBRATION: Finding h for target ρ = 2.152
============================================================

PHASE 1: Probing achievable ρ range...
----------------------------------------
  Generating HB-LFR: n=5000, h=-0.50, mu=0.30
  Target ρ_HB = 2.152 (from calibration)      ← Real target passed
  Lower bound:  h=-0.50 → ρ=0.560

  Generating HB-LFR: n=5000, h=0.00, mu=0.30
  Middle (h=0): h=0.00 → ρ=0.792              ← Tested before upper

  Generating HB-LFR: n=5000, h=3.50, mu=0.30
  Target ρ_HB = 2.152 (from calibration)
  Upper bound:  h=3.50 → ρ=2.414

  Achievable range: [0.560, 2.414]
  Target:           2.152

PHASE 2: Checking target achievability...
----------------------------------------
  Target is ACHIEVABLE within range
  Proceeding with binary search...

PHASE 3: Binary search for optimal h...
  h=1.50, Target ρ_HB = 2.152 (from calibration)
  h=1.50 → ρ=1.025 (diff=1.127)
  h=2.50 → ρ=2.203 (diff=0.051)
  h=2.38 → ρ=2.158 (diff=0.006)
  Found excellent match!

CALIBRATION RESULT
  Target ρ:      2.152
  Achievable:    [0.560, 2.414]
  Best h:        2.380
  Best ρ:        2.158
  Difference:    0.006
  Status:        SUCCESS
```

**When target is NOT achievable (upper probe succeeds):**
```
PHASE 1: Probing achievable ρ range...
----------------------------------------
  Lower bound:  h=-0.50 → ρ=0.784
  Middle (h=0): h=0.00 → ρ=0.768
  Upper bound:  h=3.50 → ρ=1.432

  Achievable range: [0.768, 1.432]
  Target:           1.484

PHASE 2: Checking target achievability...
----------------------------------------
  *** TARGET TOO HIGH ***
  Target ρ=1.484 > max achievable ρ=1.432
  Using best available: h=3.50 → ρ=1.432

PHASE 3: Binary search SKIPPED (target not achievable)

CALIBRATION RESULT
  Target ρ:      1.484
  Achievable:    [0.768, 1.432]
  Best h:        3.500
  Best ρ:        1.432
  Difference:    0.052
  Status:        TARGET NOT ACHIEVABLE
```

**When upper probe fails (estimation kicks in):**
```
PHASE 1: Probing achievable ρ range...
----------------------------------------
  Lower bound:  h=-0.50 → ρ=0.310
  Middle (h=0): h=0.00 → ρ=0.768
  Upper bound:  h=3.50 → FAILED (generation error)
  Estimating rho_upper = 0.998 (h=0 result × 1.3 for mu=0.19)
  Note: Upper bound is ESTIMATED (probe failed)

  Achievable range: [0.310, 0.998]
  Target:           1.484

PHASE 2: Checking target achievability...
----------------------------------------
  *** TARGET TOO HIGH ***
  Target ρ=1.484 > max achievable ρ=0.998
  Skipping binary search...
```

---

## Network Domains and Expected rho_HB

> **Note:** The ρ_HB values below are **estimates** that may differ from actual measurements.
> Actual ρ_HB depends on:
> 1. **Community detection algorithm** (Leiden vs Louvain vs Label Propagation)
> 2. **Resolution parameter** used in community detection
> 3. **Network preprocessing** (largest connected component, etc.)
>
> The experiment measures actual ρ_HB using Leiden algorithm at runtime.
> Some networks (e.g., ca-HepTh) may show different ρ_HB than listed here.

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
    max_nodes=100000,  # Up to 100k nodes
)

# Run extended experiment (uses binary search calibration automatically)
results = experiment_5_extended(
    real_networks_dict=networks,
    n_synthetic_per_real=50,      # Validation samples
    use_extended_h_fitting=True,  # Extended h range (-0.5, 3.5)
    use_weighted_distance=True,   # Weighted property distance
    seed=42,
    # Note: Calibration uses binary search (~18 samples per network)
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

The experiment uses **smart 3-phase calibration** which is ~10x faster than grid search:

**Speed optimizations built-in:**
1. **Phase 1 probe** (3 samples) determines achievable range upfront
2. **Phase 2 skip** - Binary search entirely skipped if target unreachable
3. **FAST MODE** - Reduced max_iters=500 (vs 5000) when target not achievable
4. **Essential params only** - No over-constraining means fewer failed LFR generations

If still slow:
- Use `--quick` for testing
- Reduce `--n-samples` (e.g., 10 instead of 50)
- Filter to smaller networks (< 10,000 nodes)
- Networks with extreme rho targets will be processed quickly via FAST MODE

### Community Detection Issues

The loader uses **Leiden algorithm** (Traag et al. 2019) as the primary method:
1. Leiden (leidenalg) - state-of-the-art, fixes Louvain's resolution issues
2. Louvain (fallback if leidenalg not installed)
3. Label Propagation (last resort fallback)

Install leidenalg for best results:
```bash
pip install leidenalg python-igraph
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
