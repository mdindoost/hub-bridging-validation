# Experiment Specifications

This document describes each validation experiment in detail.

## Structural Validation

### Experiment 1: Parameter Control

**Hypothesis**: The hub-bridging exponent h monotonically controls ρ_HB.

**Method**:
1. Generate networks at h ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
2. Compute ρ_HB for each network
3. Test monotonicity using Spearman correlation
4. Fit polynomial model

**Success Criteria**:
- Spearman r > 0.9 with p < 0.05
- Polynomial fit R² > 0.95

### Experiment 2: Degree Preservation

**Hypothesis**: The degree distribution is preserved across h values.

**Method**:
1. Generate networks at h ∈ {0.0, 0.5, 1.0}
2. Compute degree distributions
3. Compare using Kolmogorov-Smirnov test
4. Compare power-law exponents

**Success Criteria**:
- KS test p-value > 0.01 for all pairs
- Power-law exponents within 0.1 of target

### Experiment 3: Modularity Independence

**Hypothesis**: Modularity Q is controlled by μ, not h.

**Method**:
1. Generate networks at multiple (μ, h) combinations
2. Compute modularity for each
3. Test if h affects Q using Kruskal-Wallis test

**Success Criteria**:
- Kruskal-Wallis p > 0.05 for each μ level

### Experiment 4: Concentration

**Hypothesis**: ρ_HB has low variance (concentrated around mean).

**Method**:
1. Generate many samples at fixed h
2. Compute coefficient of variation (CV)
3. Test normality with Shapiro-Wilk

**Success Criteria**:
- CV < 0.1 for all h values

## Realism Validation

### Experiment 5: Property Matching

**Hypothesis**: Synthetic networks can match properties of real networks.

**Method**:
1. Compute properties of real networks
2. Generate synthetic networks at various h
3. Compute property distances
4. Find h that minimizes distance

**Metrics**:
- Hub-bridging ratio
- Clustering coefficient
- Modularity
- Path length

### Experiment 6: Network Fitting

**Hypothesis**: Optimal parameters can be found to match specific networks.

**Method**:
1. Select target real network
2. Optimize (h, μ, ...) to minimize property distance
3. Generate fitted network
4. Compare properties

## Algorithmic Validation

### Experiment 7: Community Detection

**Hypothesis**: Hub-bridging affects community detection performance.

**Method**:
1. Generate networks at various h
2. Run community detection algorithms
3. Compare detected vs ground truth using NMI/ARI
4. Analyze performance vs h

**Algorithms**:
- Louvain
- Leiden
- Label Propagation
- Infomap

### Experiment 8: Sparsification

**Hypothesis**: Hub-bridging affects sparsification behavior.

**Method**:
1. Generate networks at various h
2. Apply sparsification methods
3. Measure community preservation after sparsification
4. Analyze inter/intra edge retention

**Methods**:
- DSpar (degree-based)
- Random
- Degree-based
