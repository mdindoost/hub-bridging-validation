"""Setup script for Hub-Bridging Validation Framework."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = []

setup(
    name="hub_bridging_validation",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@university.edu",
    description="Validation framework for hub-bridging graph generators (HB-LFR, HB-SBM)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hub-bridging-validation",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/hub-bridging-validation/issues",
        "Documentation": "https://github.com/yourusername/hub-bridging-validation/docs",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4",
            "pytest-cov>=4.1",
            "black>=23.0",
            "isort>=5.12",
            "mypy>=1.5",
        ],
        "docs": [
            "sphinx>=7.0",
            "sphinx-rtd-theme>=1.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "hb-validate=experiments.run_full_validation:main",
            "hb-structural=experiments.run_structural_validation:main",
            "hb-realism=experiments.run_realism_validation:main",
            "hb-algorithmic=experiments.run_algorithmic_validation:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
