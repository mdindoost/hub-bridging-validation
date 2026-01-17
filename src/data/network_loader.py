"""
Network Loader Module
=====================

This module provides utilities for loading real network datasets
from various sources and formats.

Supported formats:
- Edge list (space, tab, or comma separated)
- NetworkX pickle files (.pkl, .pickle)
- GML files (.gml)
- GraphML files (.graphml)
- Pajek files (.net)

Expected directory structure for SNAP datasets:
    data/real_networks/
        social/
            facebook_combined.txt
            email-Eu-core.txt
        biological/
            bio-yeast.txt
        technological/
            p2p-Gnutella.txt
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np

from ..config import COMMUNITY_DETECTION_SEED, RANDOM_SEED

logger = logging.getLogger(__name__)


def load_real_networks_from_snap(
    data_dir: str = "data/real_networks",
    domains: Optional[List[str]] = None,
    max_nodes: Optional[int] = None,
    min_nodes: int = 50,
    require_communities: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Load real networks from SNAP dataset collection.

    Scans the data directory for network files organized by domain
    (social, biological, technological, etc.) and loads them into
    NetworkX graphs.

    Parameters
    ----------
    data_dir : str
        Directory containing real network data
    domains : list of str, optional
        Which domains to load. If None, loads all found domains.
    max_nodes : int, optional
        Maximum number of nodes. Networks larger than this are skipped.
    min_nodes : int
        Minimum number of nodes. Networks smaller than this are skipped.
    require_communities : bool
        If True, skip networks without ground-truth communities.

    Returns
    -------
    dict
        {
            'network_name': {
                'G': networkx.Graph,
                'domain': str (social/biological/technological/etc),
                'source': str (file path),
                'communities': list of sets (ground truth if available, else detected),
                'metadata': dict (basic network statistics)
            }
        }

    Notes
    -----
    File formats supported:
    - .txt, .edges, .csv: Edge list (auto-detects delimiter)
    - .pkl, .pickle: NetworkX pickle
    - .gml: GML format
    - .graphml: GraphML format
    - .net: Pajek format

    Community files:
    - Looks for {network_name}_communities.txt or {network_name}.cmty
    - If not found, uses Louvain community detection
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.warning(f"Data directory {data_dir} does not exist")
        return {}

    networks = {}

    # Scan for domain subdirectories or files directly
    if any(data_path.iterdir()):
        # Check if organized by domain
        subdirs = [d for d in data_path.iterdir() if d.is_dir()]

        if subdirs:
            # Organized by domain
            for domain_dir in subdirs:
                domain = domain_dir.name

                if domains is not None and domain not in domains:
                    continue

                logger.info(f"Scanning domain: {domain}")

                for file_path in domain_dir.iterdir():
                    if _is_network_file(file_path):
                        network_name = file_path.stem
                        result = _load_single_network(
                            file_path, domain, max_nodes, min_nodes
                        )
                        if result is not None:
                            networks[network_name] = result
        else:
            # All files in root directory
            domain = "unknown"
            for file_path in data_path.iterdir():
                if _is_network_file(file_path):
                    network_name = file_path.stem
                    result = _load_single_network(
                        file_path, domain, max_nodes, min_nodes
                    )
                    if result is not None:
                        networks[network_name] = result

    # Detect communities if missing (using Leiden algorithm)
    for name, data in networks.items():
        if data.get("communities") is None:
            logger.info(f"Detecting communities for {name} using Leiden algorithm")
            communities = detect_communities_if_missing(data["G"], method="leiden", seed=COMMUNITY_DETECTION_SEED)
            data["communities"] = communities
            # Validate communities
            try:
                validate_communities(data["G"], communities)
            except AssertionError as e:
                logger.warning(f"Community validation warning for {name}: {e}")

        # Skip if require_communities and still none
        if require_communities and data.get("communities") is None:
            logger.warning(f"Skipping {name}: no communities available")
            del networks[name]
            continue

        # Extract metadata
        data["metadata"] = extract_network_metadata(
            data["G"], data["communities"], name
        )

    logger.info(f"Loaded {len(networks)} networks")
    return networks


def _is_network_file(path: Path) -> bool:
    """Check if file is a supported network format."""
    supported_extensions = {
        ".txt", ".edges", ".csv", ".edgelist",
        ".pkl", ".pickle",
        ".gml", ".graphml", ".net"
    }
    return path.is_file() and path.suffix.lower() in supported_extensions


def _load_single_network(
    file_path: Path,
    domain: str,
    max_nodes: Optional[int],
    min_nodes: int,
) -> Optional[Dict[str, Any]]:
    """Load a single network file."""
    try:
        G, communities = load_network_file(str(file_path))

        if G is None:
            return None

        # Filter by size
        n = G.number_of_nodes()
        if n < min_nodes:
            logger.debug(f"Skipping {file_path.name}: too small ({n} < {min_nodes})")
            return None
        if max_nodes is not None and n > max_nodes:
            logger.debug(f"Skipping {file_path.name}: too large ({n} > {max_nodes})")
            return None

        # Get largest connected component
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            logger.debug(
                f"Using largest component: {G.number_of_nodes()} of {n} nodes"
            )

            # Update communities if they exist
            if communities is not None:
                communities = _filter_communities_to_subgraph(communities, set(G.nodes()))

        logger.info(
            f"Loaded {file_path.name}: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges"
        )

        return {
            "G": G,
            "domain": domain,
            "source": str(file_path),
            "communities": communities,
        }

    except Exception as e:
        logger.warning(f"Failed to load {file_path}: {e}")
        return None


def _filter_communities_to_subgraph(
    communities: List[Set[int]],
    nodes: Set[int],
) -> List[Set[int]]:
    """Filter communities to only include nodes in subgraph."""
    filtered = []
    for comm in communities:
        filtered_comm = comm & nodes
        if filtered_comm:
            filtered.append(filtered_comm)
    return filtered


def load_network_file(
    filepath: str,
) -> Tuple[Optional[nx.Graph], Optional[List[Set[int]]]]:
    """
    Load a network from various file formats.

    Parameters
    ----------
    filepath : str
        Path to network file

    Returns
    -------
    tuple
        (Graph, communities) where communities may be None

    Supported formats:
    - .txt, .edges, .csv, .edgelist: Edge list
    - .pkl, .pickle: NetworkX pickle
    - .gml: GML format
    - .graphml: GraphML format
    - .net: Pajek format
    """
    path = Path(filepath)
    suffix = path.suffix.lower()

    G = None
    communities = None

    try:
        if suffix in {".txt", ".edges", ".csv", ".edgelist"}:
            G = _load_edge_list(filepath)
        elif suffix in {".pkl", ".pickle"}:
            import pickle
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, nx.Graph):
                G = data
            elif isinstance(data, dict):
                G = data.get("G") or data.get("graph")
                communities = data.get("communities")
        elif suffix == ".gml":
            G = nx.read_gml(filepath)
        elif suffix == ".graphml":
            G = nx.read_graphml(filepath)
        elif suffix == ".net":
            G = nx.read_pajek(filepath)
        else:
            logger.warning(f"Unknown file format: {suffix}")
            return None, None

        # Convert to undirected simple graph
        if G is not None:
            if G.is_directed():
                G = G.to_undirected()
            if G.is_multigraph():
                G = nx.Graph(G)

        # Look for community file
        if communities is None:
            communities = _load_communities_file(path)

        return G, communities

    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None, None


def _load_edge_list(filepath: str) -> nx.Graph:
    """Load edge list with auto-detected delimiter."""
    G = nx.Graph()

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#") or line.startswith("%"):
                continue

            # Auto-detect delimiter
            if "\t" in line:
                parts = line.split("\t")
            elif "," in line:
                parts = line.split(",")
            else:
                parts = line.split()

            if len(parts) >= 2:
                try:
                    u = int(parts[0])
                    v = int(parts[1])
                    if u != v:  # Skip self-loops
                        G.add_edge(u, v)
                except ValueError:
                    # Try as strings if int conversion fails
                    u, v = parts[0], parts[1]
                    if u != v:
                        G.add_edge(u, v)

    return G


def _load_communities_file(network_path: Path) -> Optional[List[Set[int]]]:
    """Look for and load community file."""
    possible_files = [
        network_path.with_suffix(".cmty"),
        network_path.with_suffix(".communities"),
        network_path.parent / f"{network_path.stem}_communities.txt",
        network_path.parent / f"{network_path.stem}.cmty.txt",
    ]

    for comm_path in possible_files:
        if comm_path.exists():
            try:
                return _parse_community_file(str(comm_path))
            except Exception as e:
                logger.warning(f"Failed to parse {comm_path}: {e}")

    return None


def _parse_community_file(filepath: str) -> List[Set[int]]:
    """Parse community file (one community per line)."""
    communities = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Parse node IDs
            if "\t" in line:
                parts = line.split("\t")
            elif "," in line:
                parts = line.split(",")
            else:
                parts = line.split()

            members = set()
            for part in parts:
                try:
                    members.add(int(part))
                except ValueError:
                    pass

            if members:
                communities.append(members)

    return communities


def detect_communities_if_missing(
    G: nx.Graph,
    method: str = "leiden",
    seed: Optional[int] = None,
) -> List[Set[int]]:
    """
    Detect communities if ground truth not available.

    Uses Leiden algorithm by default (state-of-the-art).

    Parameters
    ----------
    G : networkx.Graph
        Input graph
    method : str
        Ignored (always uses Leiden). Kept for API compatibility.
    seed : int, optional
        Random seed for reproducibility (default: COMMUNITY_DETECTION_SEED)

    Returns
    -------
    list of sets
        Detected communities as list of node sets
    """
    if seed is None:
        seed = COMMUNITY_DETECTION_SEED

    if G.number_of_nodes() == 0:
        return []

    # Always use Leiden (state-of-the-art) with fallback
    return detect_communities_robust(G, method="leiden", fallback=True, seed=seed)


def extract_network_metadata(
    G: nx.Graph,
    communities: Optional[List[Set[int]]],
    network_name: str,
) -> Dict[str, Any]:
    """
    Extract basic metadata from network.

    Parameters
    ----------
    G : networkx.Graph
        Input graph
    communities : list of sets
        Community structure
    network_name : str
        Name of the network

    Returns
    -------
    dict
        Basic network statistics
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    metadata = {
        "name": network_name,
        "n": n,
        "m": m,
        "density": nx.density(G),
        "avg_degree": 2 * m / n if n > 0 else 0,
        "is_connected": nx.is_connected(G),
    }

    if communities is not None:
        metadata["num_communities"] = len(communities)
        comm_sizes = [len(c) for c in communities]
        metadata["avg_community_size"] = np.mean(comm_sizes) if comm_sizes else 0
        metadata["min_community_size"] = min(comm_sizes) if comm_sizes else 0
        metadata["max_community_size"] = max(comm_sizes) if comm_sizes else 0

    return metadata


def create_sample_networks(
    output_dir: str = "data/real_networks",
    include_karate: bool = True,
    include_dolphins: bool = True,
    include_football: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Create sample networks for testing using NetworkX built-in datasets.

    Parameters
    ----------
    output_dir : str
        Directory to save networks
    include_karate : bool
        Include Zachary's karate club
    include_dolphins : bool
        Include dolphins social network
    include_football : bool
        Include American football network

    Returns
    -------
    dict
        Dictionary of loaded networks
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    networks = {}

    if include_karate:
        G = nx.karate_club_graph()
        # Ground truth communities
        communities = [
            {0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 19, 21},
            {8, 9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33},
        ]
        networks["karate"] = {
            "G": G,
            "domain": "social",
            "source": "networkx_builtin",
            "communities": communities,
            "metadata": extract_network_metadata(G, communities, "karate"),
        }

    if include_dolphins:
        try:
            G = nx.Graph()
            # Dolphins network edge list (simplified version)
            # In practice, would load from file
            logger.info("Dolphins network requires external data file")
        except Exception:
            pass

    if include_football:
        try:
            # Football network (if available)
            logger.info("Football network requires external data file")
        except Exception:
            pass

    logger.info(f"Created {len(networks)} sample networks")
    return networks


# Expected ρ_HB values from previous paper
EXPECTED_RHO_HB = {
    'wiki-Talk': 10.19,
    'wiki-topcats': 4.96,
    'com-Youtube': 4.78,
    'email-Enron': 3.12,
    'facebook-combined': 2.69,
    'facebook_combined': 2.69,  # Alternative naming
    'ca-CondMat': 2.43,
    'email-Eu-core': 2.09,
    'email-Eu-core': 2.09,
    'cit-HepTh': 2.02,
    'wiki-Vote': 1.96,
    'com-DBLP': 1.62,
    'ca-HepTh': 1.54,
    'ca-AstroPh': 1.45,
    'cit-HepPh': 1.30,
    'com-Amazon': 1.03,
    'cit-Patents': 0.72,
    'ca-HepPh': 0.65,
    'ca-GrQc': 0.48,
}


def get_expected_rho(network_name: str) -> Optional[float]:
    """
    Get expected ρ_HB from previous paper's table.

    Parameters
    ----------
    network_name : str
        Name of the network

    Returns
    -------
    float or None
        Expected ρ_HB if known, else None
    """
    # Try exact match first
    if network_name in EXPECTED_RHO_HB:
        return EXPECTED_RHO_HB[network_name]

    # Try with common transformations
    transformed = network_name.replace('_', '-').replace('.txt', '')
    if transformed in EXPECTED_RHO_HB:
        return EXPECTED_RHO_HB[transformed]

    # Try without extension
    stem = Path(network_name).stem
    if stem in EXPECTED_RHO_HB:
        return EXPECTED_RHO_HB[stem]

    return None


def detect_communities_leiden(
    G: nx.Graph,
    seed: Optional[int] = None,
) -> List[Set[int]]:
    """
    Detect communities using Leiden algorithm.

    Leiden is the state-of-the-art modularity optimization algorithm,
    superior to Louvain (fixes disconnected communities issue).

    Citation: Traag et al. (2019) Scientific Reports 9:5233
    "From Louvain to Leiden: guaranteeing well-connected communities"

    Parameters
    ----------
    G : networkx.Graph
        Input graph
    seed : int, optional
        Random seed for reproducibility (default: COMMUNITY_DETECTION_SEED)

    Returns
    -------
    list of sets
        Communities (each community is a set of node IDs)

    Raises
    ------
    ImportError
        If leidenalg not installed
    """
    if seed is None:
        seed = COMMUNITY_DETECTION_SEED

    if G.number_of_nodes() == 0:
        return []

    try:
        import leidenalg
        import igraph as ig
    except ImportError:
        raise ImportError(
            "Leiden algorithm requires: pip install leidenalg python-igraph"
        )

    # Convert NetworkX graph to igraph
    # Map node IDs to integers (igraph requirement)
    node_list = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]

    g_igraph = ig.Graph(n=len(node_list), edges=edges, directed=False)

    # Run Leiden with fixed seed for reproducibility
    partition = leidenalg.find_partition(
        g_igraph,
        leidenalg.ModularityVertexPartition,
        seed=seed,
        n_iterations=-1  # Run until convergence
    )

    # Convert back to NetworkX node IDs
    communities = []
    for comm_nodes in partition:
        community = {node_list[idx] for idx in comm_nodes}
        communities.append(community)

    logger.info(f"Leiden found {len(communities)} communities (Q={partition.quality():.4f})")

    return communities


def validate_communities(
    G: nx.Graph,
    communities: List[Set[int]],
) -> bool:
    """
    Validate that communities are well-formed.

    Checks:
    - No empty communities
    - All nodes covered
    - No overlaps (for non-overlapping detection)
    - Modularity > 0

    Parameters
    ----------
    G : networkx.Graph
        Input graph
    communities : list of sets
        Detected communities

    Returns
    -------
    bool
        True if validation passes

    Raises
    ------
    AssertionError
        If any validation check fails
    """
    # Check coverage
    all_nodes = set()
    for comm in communities:
        all_nodes.update(comm)

    if all_nodes != set(G.nodes()):
        missing = set(G.nodes()) - all_nodes
        extra = all_nodes - set(G.nodes())
        raise AssertionError(
            f"Communities don't cover all nodes. Missing: {len(missing)}, Extra: {len(extra)}"
        )

    # Check no empty
    empty_comms = [i for i, c in enumerate(communities) if len(c) == 0]
    if empty_comms:
        raise AssertionError(f"Empty communities found at indices: {empty_comms}")

    # Check no overlaps
    total_in_comms = sum(len(c) for c in communities)
    if total_in_comms != len(all_nodes):
        raise AssertionError(
            f"Overlapping communities detected: {total_in_comms} total vs {len(all_nodes)} unique"
        )

    # Check modularity
    Q = nx.community.modularity(G, communities)
    if Q <= 0:
        logger.warning(f"Low modularity: Q={Q:.4f} (may indicate poor community structure)")

    return True


def detect_communities_robust(
    G: nx.Graph,
    method: str = "leiden",
    fallback: bool = True,
    seed: Optional[int] = None,
) -> List[Set[int]]:
    """
    Detect communities with Leiden as primary method, with fallbacks.

    DEPRECATED: Use detect_communities_leiden() directly.
    This function is kept for backwards compatibility.

    Try in order:
    1. Leiden (state-of-the-art, fixes Louvain issues)
    2. Louvain (NetworkX built-in)
    3. Label Propagation (built-in, fast)
    4. Connected components (last resort)

    Parameters
    ----------
    G : networkx.Graph
        Input graph
    method : str
        Ignored (always uses Leiden first)
    fallback : bool
        Whether to try fallback methods
    seed : int, optional
        Random seed

    Returns
    -------
    list of sets
        Detected communities
    """
    if seed is None:
        seed = COMMUNITY_DETECTION_SEED

    if G.number_of_nodes() == 0:
        return []

    # Try Leiden first (state-of-the-art)
    try:
        communities = detect_communities_leiden(G, seed=seed)
        validate_communities(G, communities)
        return communities
    except ImportError:
        logger.warning("leidenalg not installed. Install with: pip install leidenalg python-igraph")
    except Exception as e:
        logger.warning(f"Leiden failed: {e}")

    if not fallback:
        raise RuntimeError("Leiden algorithm failed and fallback=False")

    # Fallback: Try NetworkX Louvain
    try:
        communities = list(nx.community.louvain_communities(G, seed=seed, resolution=1.0))
        logger.warning("Falling back to Louvain (Leiden preferred)")
        return communities
    except Exception as e:
        logger.debug(f"NetworkX Louvain failed: {e}")

    # Fallback: Try label propagation (built-in, fast)
    try:
        communities = list(nx.community.label_propagation_communities(G))
        communities = [set(c) for c in communities]
        logger.warning("Falling back to Label Propagation")
        return communities
    except Exception as e:
        logger.debug(f"Label propagation failed: {e}")

    # Last resort: connected components
    logger.error("All community detection methods failed, using connected components")
    communities = [set(c) for c in nx.connected_components(G)]
    return communities


def load_networks_for_experiment_5(
    data_dir: str = "data/real_networks",
    min_nodes: int = 100,
    max_nodes: int = 100000,
    domains: Optional[List[str]] = None,
    include_expected_rho: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Load networks specifically formatted for Experiment 5.

    This is a convenience wrapper around load_real_networks_from_snap
    that adds expected ρ_HB values and ensures proper formatting.

    Parameters
    ----------
    data_dir : str
        Directory containing network files
    min_nodes : int
        Minimum network size
    max_nodes : int
        Maximum network size (larger networks are subsampled)
    domains : list of str, optional
        Filter by domain
    include_expected_rho : bool
        Whether to add expected ρ_HB to metadata

    Returns
    -------
    dict
        Networks formatted for Experiment 5
    """
    networks = load_real_networks_from_snap(
        data_dir=data_dir,
        domains=domains,
        max_nodes=max_nodes,
        min_nodes=min_nodes,
        require_communities=False,
    )

    # Add expected ρ_HB values
    if include_expected_rho:
        for name, data in networks.items():
            expected_rho = get_expected_rho(name)
            data["metadata"]["expected_rho_HB"] = expected_rho

            if expected_rho is not None:
                logger.info(f"  {name}: expected ρ_HB = {expected_rho:.2f}")
            else:
                logger.info(f"  {name}: expected ρ_HB = unknown")

    return networks
