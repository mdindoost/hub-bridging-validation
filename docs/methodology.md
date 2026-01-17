# Methodology

This document describes the methodology used in the hub-bridging validation framework.

## Hub-Bridging Ratio (ρ_HB)

### Definition

The hub-bridging ratio quantifies the relationship between node degree and edge placement:

```
ρ_HB = E[d_u × d_v | (u,v) is inter-community] / E[d_u × d_v | (u,v) is intra-community]
```

Where:
- `d_u`, `d_v` are the degrees of nodes u and v
- Inter-community edges connect nodes in different communities
- Intra-community edges connect nodes in the same community

### Interpretation

- **ρ_HB > 1**: High-degree nodes (hubs) preferentially form inter-community (bridging) edges
- **ρ_HB = 1**: No degree-based preference for edge placement
- **ρ_HB < 1**: High-degree nodes preferentially form intra-community edges

## HB-LFR Generator

### Algorithm (Rewiring-based)

1. Generate standard LFR benchmark graph
2. Classify edges as inter/intra-community
3. Rewire edges to increase hub-bridging:
   - Select low-score inter-community edge
   - Select high-score inter-community edge
   - Swap if valid (preserves edge types)
4. Repeat until convergence or max iterations

### Hub-Bridging Exponent (h)

The parameter h controls the strength of hub-bridging preference:
- h = 0: Standard LFR behavior
- h > 0: Increased hub-bridging (higher ρ_HB)

Edge selection probability is proportional to `(d_u × d_v)^h`.

## HB-SBM Generator

### Model

For nodes u, v with expected degrees θ_u, θ_v:

```
P(edge u,v) = {
  p_in × θ_u × θ_v,                if same community
  p_out × (θ_u × θ_v)^(1+h),       if different communities
}
```

This directly incorporates hub-bridging into the edge probability model.

## Validation Approach

### Statistical Framework

All experiments use:
- Multiple samples per parameter setting (default: 30)
- 95% confidence intervals
- Multiple testing corrections (Bonferroni or FDR)
- Effect size calculations (Cohen's d)

### Experiment Design

Each experiment tests a specific hypothesis about generator behavior:
1. **Positive control**: Generator should exhibit expected behavior
2. **Negative control**: Generator should NOT exhibit undesired behavior
3. **Comparison**: Generator should match/outperform baselines
