# API Reference

## Metrics Module

### `src.metrics.hub_bridging`

#### `compute_hub_bridging_ratio(G, communities, weighted=False)`
Compute the hub-bridging ratio ρ_HB.

**Parameters:**
- `G` (nx.Graph): Input graph
- `communities` (Dict[int, int]): Node to community mapping
- `weighted` (bool): Use weighted degrees

**Returns:** float - Hub-bridging ratio

#### `compute_dspar_separation(G, communities, normalize=True)`
Compute DSpar separation metric (Cohen's d for DSpar scores).

**Returns:** float - Separation metric

### `src.metrics.network_properties`

#### `comprehensive_network_properties(G, communities=None, compute_expensive=True)`
Compute comprehensive network property dictionary.

**Returns:** Dict with keys: basic, degree, clustering, path_length, community, hub_bridging

#### `compute_participation_coefficient(G, communities)`
Compute participation coefficient for all nodes.

**Returns:** NDArray - Participation coefficients

### `src.metrics.distance_metrics`

#### `maximum_mean_discrepancy(X, Y, kernel='rbf', gamma=None)`
Compute MMD between two distributions.

**Returns:** float - MMD distance

## Generators Module

### `src.generators.hb_lfr`

#### `hb_lfr(n, tau1, tau2, mu, h=0.0, **kwargs)`
Generate Hub-Bridging LFR benchmark graph.

**Parameters:**
- `n` (int): Number of nodes
- `tau1` (float): Degree distribution exponent
- `tau2` (float): Community size exponent
- `mu` (float): Mixing parameter
- `h` (float): Hub-bridging exponent

**Returns:** Tuple[nx.Graph, Dict[int, int]] - (graph, communities)

### `src.generators.hb_sbm`

#### `hb_sbm(n, k, p_in, p_out, h=0.0, **kwargs)`
Generate Hub-Bridging SBM graph.

**Returns:** Tuple[nx.Graph, Dict[int, int]]

### `src.generators.calibration`

#### `calibrate_h_to_rho(generator_func, generator_params, h_values, n_samples, seed)`
Calibrate h → ρ_HB relationship.

**Returns:** Dict with calibration results and interpolator

## Validation Module

### `src.validation.structural`

#### `experiment_1_parameter_control(generator_func, generator_params, h_values, n_samples, seed)`
Run Experiment 1: Parameter Control.

**Returns:** Dict with rho samples, statistics, and test results

### `src.validation.realism`

#### `experiment_5_property_matching(real_networks, generator_func, generator_params, h_values, n_samples, seed)`
Run Experiment 5: Property Matching.

**Returns:** Dict with property comparisons and best h values

## Algorithms Module

### `src.algorithms.community_detection`

#### `detect_communities(G, algorithm='louvain', **kwargs)`
Detect communities using specified algorithm.

**Returns:** Dict[int, int] - Node to community mapping

#### `compute_nmi(true_communities, detected_communities, nodes=None)`
Compute Normalized Mutual Information.

**Returns:** float - NMI score

### `src.algorithms.sparsification`

#### `sparsify_graph(G, method='dspar', ratio=0.5, seed=None)`
Sparsify graph using specified method.

**Returns:** nx.Graph - Sparsified graph

## Visualization Module

### `src.visualization.plots`

#### `plot_rho_vs_h(h_values, rho_samples, fit_coeffs=None, ...)`
Plot ρ_HB vs h with optional polynomial fit.

**Returns:** plt.Figure

#### `plot_algorithm_performance(results, metric='nmi')`
Plot community detection performance.

**Returns:** plt.Figure
