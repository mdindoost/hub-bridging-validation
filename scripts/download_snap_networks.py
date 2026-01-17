#!/usr/bin/env python3
"""
Download Networks from SNAP and Other Sources
==============================================

Downloads 21 networks across 5 domains for Experiment 5:
- Social (6 networks)
- Collaboration (7 networks)
- Citation (3 networks)
- Communication (3 networks)
- Web (1 network)
- E-commerce (1 network)

Usage:
    python scripts/download_snap_networks.py
    python scripts/download_snap_networks.py --tier 1  # Only original 17
    python scripts/download_snap_networks.py --quick    # Small subset for testing
"""

import argparse
import gzip
import logging
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Expected ρ_HB values from previous paper
EXPECTED_RHO = {
    'wiki-Talk': 10.19,
    'wiki-topcats': 4.96,
    'com-Youtube': 4.78,
    'email-Enron': 3.12,
    'facebook-combined': 2.69,
    'ca-CondMat': 2.43,
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


# Network definitions with URLs and metadata
TIER_1_NETWORKS = {
    # SOCIAL NETWORKS
    'facebook-combined': {
        'url': 'https://snap.stanford.edu/data/facebook_combined.txt.gz',
        'domain': 'social',
        'expected_rho': 2.69,
    },
    'wiki-Vote': {
        'url': 'https://snap.stanford.edu/data/wiki-Vote.txt.gz',
        'domain': 'social',
        'expected_rho': 1.96,
    },
    'wiki-Talk': {
        'url': 'https://snap.stanford.edu/data/wiki-Talk.txt.gz',
        'domain': 'social',
        'expected_rho': 10.19,  # EXTREME
    },
    'com-Youtube': {
        'url': 'https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz',
        'domain': 'social',
        'expected_rho': 4.78,
    },

    # COMMUNICATION NETWORKS
    'email-Enron': {
        'url': 'https://snap.stanford.edu/data/email-Enron.txt.gz',
        'domain': 'communication',
        'expected_rho': 3.12,
    },
    'email-Eu-core': {
        'url': 'https://snap.stanford.edu/data/email-Eu-core.txt.gz',
        'domain': 'communication',
        'expected_rho': 2.09,
    },

    # COLLABORATION NETWORKS
    'ca-CondMat': {
        'url': 'https://snap.stanford.edu/data/ca-CondMat.txt.gz',
        'domain': 'collaboration',
        'expected_rho': 2.43,
    },
    'ca-HepTh': {
        'url': 'https://snap.stanford.edu/data/ca-HepTh.txt.gz',
        'domain': 'collaboration',
        'expected_rho': 1.54,
    },
    'ca-AstroPh': {
        'url': 'https://snap.stanford.edu/data/ca-AstroPh.txt.gz',
        'domain': 'collaboration',
        'expected_rho': 1.45,
    },
    'ca-HepPh': {
        'url': 'https://snap.stanford.edu/data/ca-HepPh.txt.gz',
        'domain': 'collaboration',
        'expected_rho': 0.65,  # HUB-ISOLATION
    },
    'ca-GrQc': {
        'url': 'https://snap.stanford.edu/data/ca-GrQc.txt.gz',
        'domain': 'collaboration',
        'expected_rho': 0.48,  # EXTREME HUB-ISOLATION
    },
    'com-DBLP': {
        'url': 'https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz',
        'domain': 'collaboration',
        'expected_rho': 1.62,
    },
    'wiki-topcats': {
        'url': 'https://snap.stanford.edu/data/wiki-topcats.txt.gz',
        'domain': 'collaboration',
        'expected_rho': 4.96,
    },

    # CITATION NETWORKS
    'cit-HepTh': {
        'url': 'https://snap.stanford.edu/data/cit-HepTh.txt.gz',
        'domain': 'citation',
        'expected_rho': 2.02,
    },
    'cit-HepPh': {
        'url': 'https://snap.stanford.edu/data/cit-HepPh.txt.gz',
        'domain': 'citation',
        'expected_rho': 1.30,
    },
    'cit-Patents': {
        'url': 'https://snap.stanford.edu/data/cit-Patents.txt.gz',
        'domain': 'citation',
        'expected_rho': 0.72,  # HUB-ISOLATION
    },

    # E-COMMERCE
    'com-Amazon': {
        'url': 'https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz',
        'domain': 'ecommerce',
        'expected_rho': 1.03,
    },
}


TIER_2_NETWORKS = {
    # ADDITIONAL SOCIAL NETWORKS
    'soc-Epinions': {
        'url': 'https://snap.stanford.edu/data/soc-Epinions1.txt.gz',
        'domain': 'social',
        'expected_rho': None,
    },
    'soc-Slashdot': {
        'url': 'https://snap.stanford.edu/data/soc-Slashdot0902.txt.gz',
        'domain': 'social',
        'expected_rho': None,
    },
    'web-Google': {
        'url': 'https://snap.stanford.edu/data/web-Google.txt.gz',
        'domain': 'web',
        'expected_rho': None,
    },

    # ADDITIONAL COMMUNICATION
    'email-EuAll': {
        'url': 'https://snap.stanford.edu/data/email-EuAll.txt.gz',
        'domain': 'communication',
        'expected_rho': None,
    },

    # NOTE: Removed networks with broken URLs:
    # - bio-yeast (ppi-yeast.txt.gz not available)
    # - as-733 (as-733.txt.gz not available)
}


# Quick test subset (small networks for fast testing)
QUICK_TEST_NETWORKS = [
    'ca-GrQc',       # Small, hub-isolation
    'email-Eu-core', # Small, moderate
    'ca-HepTh',      # Medium, moderate
]


def get_domain_from_name(name: str) -> str:
    """Determine domain from network name."""
    if name.startswith('bio-'):
        return 'biological'
    elif name.startswith('road') or name.startswith('power') or name.startswith('as-'):
        return 'infrastructure'
    elif name.startswith('soc-') or name.startswith('facebook') or name.startswith('wiki'):
        return 'social'
    elif name.startswith('email'):
        return 'communication'
    elif name.startswith('web-'):
        return 'web'
    elif name.startswith('ca-') or name.startswith('com-DBLP'):
        return 'collaboration'
    elif name.startswith('cit-'):
        return 'citation'
    elif name.startswith('com-'):
        return 'ecommerce'
    else:
        return 'other'


def download_network(
    name: str,
    url: str,
    output_dir: str,
    domain: Optional[str] = None,
) -> bool:
    """
    Download and extract network from URL.

    Parameters
    ----------
    name : str
        Network name
    url : str
        Download URL
    output_dir : str
        Base output directory
    domain : str, optional
        Domain subdirectory

    Returns
    -------
    bool
        True if successful
    """
    if domain is None:
        domain = get_domain_from_name(name)

    # Create domain directory
    domain_dir = Path(output_dir) / domain
    domain_dir.mkdir(parents=True, exist_ok=True)

    # Determine filename
    filename = url.split('/')[-1]
    download_path = domain_dir / filename

    # Output file (after extraction)
    if filename.endswith('.gz'):
        output_file = domain_dir / filename[:-3]
    else:
        output_file = domain_dir / filename

    # Check if already exists
    if output_file.exists():
        logger.info(f"  {name}: Already exists at {output_file}")
        return True

    logger.info(f"  {name}: Downloading from {url}")

    try:
        # Download with user agent (some servers block default urllib)
        request = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; NetworkDownloader/1.0)'}
        )

        with urllib.request.urlopen(request, timeout=120) as response:
            with open(download_path, 'wb') as f:
                shutil.copyfileobj(response, f)

        # Extract if gzipped
        if filename.endswith('.gz'):
            logger.info(f"  {name}: Extracting...")
            with gzip.open(download_path, 'rb') as f_in:
                with open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(download_path)

        logger.info(f"  {name}: Saved to {output_file}")
        return True

    except urllib.error.HTTPError as e:
        logger.error(f"  {name}: HTTP Error {e.code} - {e.reason}")
        return False
    except urllib.error.URLError as e:
        logger.error(f"  {name}: URL Error - {e.reason}")
        return False
    except Exception as e:
        logger.error(f"  {name}: Error - {e}")
        # Clean up partial downloads
        if download_path.exists():
            os.remove(download_path)
        return False


def download_all_networks(
    output_dir: str = "data/real_networks",
    tier: int = 0,  # 0=all, 1=tier1 only, 2=tier2 only
    quick: bool = False,
    networks_list: Optional[list] = None,
) -> Dict[str, bool]:
    """
    Download all specified networks.

    Parameters
    ----------
    output_dir : str
        Output directory
    tier : int
        Which tier to download (0=all, 1=tier1, 2=tier2)
    quick : bool
        Only download quick test subset
    networks_list : list, optional
        Specific networks to download

    Returns
    -------
    dict
        Results for each network (name -> success)
    """
    # Determine which networks to download
    if networks_list is not None:
        all_networks = {**TIER_1_NETWORKS, **TIER_2_NETWORKS}
        networks = {name: all_networks[name] for name in networks_list if name in all_networks}
    elif quick:
        networks = {name: TIER_1_NETWORKS[name] for name in QUICK_TEST_NETWORKS}
    elif tier == 1:
        networks = TIER_1_NETWORKS
    elif tier == 2:
        networks = TIER_2_NETWORKS
    else:
        networks = {**TIER_1_NETWORKS, **TIER_2_NETWORKS}

    logger.info(f"Downloading {len(networks)} networks to {output_dir}")
    logger.info("=" * 60)

    results = {}

    for name, info in networks.items():
        success = download_network(
            name=name,
            url=info['url'],
            output_dir=output_dir,
            domain=info.get('domain'),
        )
        results[name] = success

    # Summary
    success_count = sum(results.values())
    logger.info("=" * 60)
    logger.info(f"Download complete: {success_count}/{len(networks)} successful")

    if success_count < len(networks):
        failed = [name for name, success in results.items() if not success]
        logger.warning(f"Failed downloads: {failed}")

    return results


def create_metadata_file(output_dir: str = "data/real_networks"):
    """Create a metadata file with expected ρ_HB values."""
    import json

    metadata = {
        'expected_rho_HB': EXPECTED_RHO,
        'tier_1_networks': list(TIER_1_NETWORKS.keys()),
        'tier_2_networks': list(TIER_2_NETWORKS.keys()),
        'domains': {},
    }

    # Group by domain
    for name, info in {**TIER_1_NETWORKS, **TIER_2_NETWORKS}.items():
        domain = info.get('domain', 'other')
        if domain not in metadata['domains']:
            metadata['domains'][domain] = []
        metadata['domains'][domain].append(name)

    metadata_path = Path(output_dir) / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Created metadata file: {metadata_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download SNAP networks for Experiment 5"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/real_networks",
        help="Output directory",
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Tier to download (0=all, 1=original 17, 2=new additions)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Download only quick test subset (3 small networks)",
    )
    parser.add_argument(
        "--networks",
        type=str,
        nargs="+",
        default=None,
        help="Specific networks to download",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available networks and exit",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.list:
        print("\nTIER 1 NETWORKS (Original 17):")
        print("-" * 40)
        for name, info in TIER_1_NETWORKS.items():
            rho = info.get('expected_rho', 'N/A')
            print(f"  {name:<25} ρ_HB={rho}")

        print("\nTIER 2 NETWORKS (New additions):")
        print("-" * 40)
        for name, info in TIER_2_NETWORKS.items():
            rho = info.get('expected_rho', 'N/A')
            print(f"  {name:<25} ρ_HB={rho}")

        print(f"\nQuick test subset: {QUICK_TEST_NETWORKS}")
        return

    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Download networks
    results = download_all_networks(
        output_dir=args.output,
        tier=args.tier,
        quick=args.quick,
        networks_list=args.networks,
    )

    # Create metadata file
    create_metadata_file(args.output)

    # Return exit code based on results
    if all(results.values()):
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
