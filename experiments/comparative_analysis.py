"""
Comparative Analysis: HB-SBM vs HB-LFR
======================================

Publication-quality comparison of both hub-bridging generators.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generators.hb_sbm import hb_sbm
from src.generators.hb_lfr import hb_lfr
from src.metrics.hub_bridging import compute_hub_bridging_ratio
from scipy.stats import spearmanr, pearsonr


def run_experiment_1(generator_func, generator_name, h_values, n_samples=10,
                     base_params=None, seed=42):
    """
    Run Experiment 1 (parameter control) for a generator.
    """
    print(f"\n{'='*60}")
    print(f"Running Experiment 1 for {generator_name}")
    print(f"{'='*60}")

    rng = np.random.default_rng(seed)

    rho_samples = {h: [] for h in h_values}

    for h in h_values:
        print(f"\n  h = {h:.2f}:", end=" ")

        for i in range(n_samples):
            sample_seed = int(rng.integers(0, 2**31))

            try:
                params = base_params.copy() if base_params else {}
                params['h'] = h
                params['seed'] = sample_seed

                G, communities = generator_func(**params)
                rho = compute_hub_bridging_ratio(G, communities)
                rho_samples[h].append(rho)
                print(".", end="", flush=True)

            except Exception as e:
                print(f"x", end="", flush=True)
                rho_samples[h].append(np.nan)

        valid = [r for r in rho_samples[h] if not np.isnan(r)]
        if valid:
            print(f" ρ = {np.mean(valid):.3f} ± {np.std(valid):.3f}")

    # Compute statistics
    rho_means = {h: np.nanmean(rho_samples[h]) for h in h_values}
    rho_stds = {h: np.nanstd(rho_samples[h]) for h in h_values}
    rho_cis = {h: 1.96 * rho_stds[h] / np.sqrt(n_samples) for h in h_values}

    # Flatten for correlation
    all_h = []
    all_rho = []
    for h in h_values:
        for rho in rho_samples[h]:
            if not np.isnan(rho):
                all_h.append(h)
                all_rho.append(rho)

    spearman_r, spearman_p = spearmanr(all_h, all_rho)
    pearson_r, pearson_p = pearsonr(all_h, all_rho)

    print(f"\n  Spearman r = {spearman_r:.3f}, p = {spearman_p:.2e}")
    print(f"  Pearson r = {pearson_r:.3f}, p = {pearson_p:.2e}")

    return {
        'h_values': h_values,
        'rho_samples': rho_samples,
        'rho_means': rho_means,
        'rho_stds': rho_stds,
        'rho_cis': rho_cis,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'generator': generator_name,
    }


def compare_generators():
    """
    Load or generate and compare HB-SBM vs HB-LFR results.
    """
    results_dir = Path(__file__).parent.parent / 'data' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    sbm_path = results_dir / 'experiment_1_hb_sbm.pkl'
    lfr_path = results_dir / 'experiment_1_hb_lfr.pkl'

    # Use same h values for fair comparison
    h_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    n_samples = 10

    # Load or generate HB-SBM results
    if sbm_path.exists():
        print("Loading existing HB-SBM results...")
        with open(sbm_path, 'rb') as f:
            sbm_results = pickle.load(f)
    else:
        print("Generating HB-SBM results...")
        sbm_params = {
            'n': 500,
            'k': 5,
            'p_in': 0.3,
            'p_out': 0.05,
            'theta_distribution': 'power_law',
            'degree_correction_scale': 1.5,
        }
        sbm_results = run_experiment_1(
            hb_sbm, 'HB-SBM', h_values, n_samples, sbm_params, seed=42
        )
        with open(sbm_path, 'wb') as f:
            pickle.dump(sbm_results, f)
        print(f"Saved HB-SBM results to {sbm_path}")

    # Load or generate HB-LFR results
    if lfr_path.exists():
        print("Loading existing HB-LFR results...")
        with open(lfr_path, 'rb') as f:
            lfr_results = pickle.load(f)
    else:
        print("Generating HB-LFR results...")
        lfr_params = {
            'n': 500,
            'mu': 0.3,
            'tau1': 2.5,
            'tau2': 1.5,
        }
        lfr_results = run_experiment_1(
            hb_lfr, 'HB-LFR', h_values, n_samples, lfr_params, seed=42
        )
        with open(lfr_path, 'wb') as f:
            pickle.dump(lfr_results, f)
        print(f"Saved HB-LFR results to {lfr_path}")

    # Extract comparison data
    # Handle both dict and list formats for means/stds/cis
    def get_values(results, key):
        data = results[key]
        h_vals = results['h_values']
        if isinstance(data, dict):
            return [data[h] for h in h_vals]
        else:
            return list(data)

    comparison = {
        'h_values': sbm_results['h_values'],
        'sbm': {
            'means': get_values(sbm_results, 'rho_means'),
            'stds': get_values(sbm_results, 'rho_stds'),
            'cis': get_values(sbm_results, 'rho_cis'),
            'r': sbm_results['spearman_r'],
            'p': sbm_results['spearman_p'],
        },
        'lfr': {
            'means': get_values(lfr_results, 'rho_means'),
            'stds': get_values(lfr_results, 'rho_stds'),
            'cis': get_values(lfr_results, 'rho_cis'),
            'r': lfr_results['spearman_r'],
            'p': lfr_results['spearman_p'],
        }
    }

    return comparison


def plot_comprehensive_comparison(comparison, save_path=None):
    """
    Create multi-panel comparison figure.
    """
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    h_values = comparison['h_values']

    # Panel A: Main comparison with error bars
    ax1 = fig.add_subplot(gs[0, :2])

    # HB-SBM
    ax1.errorbar(h_values, comparison['sbm']['means'],
                yerr=comparison['sbm']['cis'],
                fmt='o-', capsize=5, capthick=2, markersize=8,
                label=f"HB-SBM (r={comparison['sbm']['r']:.3f})",
                color='#2E86AB', linewidth=2, alpha=0.8)

    # HB-LFR
    ax1.errorbar(h_values, comparison['lfr']['means'],
                yerr=comparison['lfr']['cis'],
                fmt='s-', capsize=5, capthick=2, markersize=8,
                label=f"HB-LFR (r={comparison['lfr']['r']:.3f})",
                color='#A23B72', linewidth=2, alpha=0.8)

    # Reference line at ρ=1
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5,
                label='ρ=1 (no hub-bridging)')

    ax1.set_xlabel('Hub-bridging parameter h', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Hub-bridging ratio ρ_HB', fontsize=14, fontweight='bold')
    ax1.set_title('Panel A: Parameter Control Comparison',
                  fontsize=16, fontweight='bold', pad=20)
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.1, 2.1)

    # Panel B: Confidence interval comparison
    ax2 = fig.add_subplot(gs[0, 2])

    ci_sbm = comparison['sbm']['cis']
    ci_lfr = comparison['lfr']['cis']

    x = np.arange(len(h_values))
    width = 0.35

    ax2.bar(x - width/2, ci_sbm, width, label='HB-SBM',
            color='#2E86AB', alpha=0.7)
    ax2.bar(x + width/2, ci_lfr, width, label='HB-LFR',
            color='#A23B72', alpha=0.7)

    ax2.set_xlabel('h value', fontsize=12, fontweight='bold')
    ax2.set_ylabel('95% CI width', fontsize=12, fontweight='bold')
    ax2.set_title('Panel B: Precision\nComparison',
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{h:.2f}' for h in h_values], rotation=45)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel C: Residuals from linear fit
    ax3 = fig.add_subplot(gs[1, 0])

    # Fit linear models
    from numpy.polynomial import Polynomial

    p_sbm = Polynomial.fit(h_values, comparison['sbm']['means'], deg=1)
    p_lfr = Polynomial.fit(h_values, comparison['lfr']['means'], deg=1)

    residuals_sbm = np.array(comparison['sbm']['means']) - p_sbm(h_values)
    residuals_lfr = np.array(comparison['lfr']['means']) - p_lfr(h_values)

    ax3.plot(h_values, residuals_sbm, 'o-', label='HB-SBM',
            color='#2E86AB', linewidth=2, markersize=6)
    ax3.plot(h_values, residuals_lfr, 's-', label='HB-LFR',
            color='#A23B72', linewidth=2, markersize=6)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    ax3.set_xlabel('h', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Residual from linear fit', fontsize=12, fontweight='bold')
    ax3.set_title('Panel C: Linearity', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Panel D: Saturation analysis
    ax4 = fig.add_subplot(gs[1, 1])

    # Compute incremental gains
    delta_sbm = np.diff(comparison['sbm']['means'])
    delta_lfr = np.diff(comparison['lfr']['means'])
    h_mid = [(h_values[i] + h_values[i+1])/2 for i in range(len(h_values)-1)]

    ax4.plot(h_mid, delta_sbm, 'o-', label='HB-SBM',
            color='#2E86AB', linewidth=2, markersize=6)
    ax4.plot(h_mid, delta_lfr, 's-', label='HB-LFR',
            color='#A23B72', linewidth=2, markersize=6)

    ax4.set_xlabel('h', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Δρ_HB (incremental gain)', fontsize=12, fontweight='bold')
    ax4.set_title('Panel D: Saturation\nAnalysis', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

    # Panel E: Statistical summary table
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')

    table_data = [
        ['Metric', 'HB-SBM', 'HB-LFR'],
        ['Spearman r', f"{comparison['sbm']['r']:.3f}",
         f"{comparison['lfr']['r']:.3f}"],
        ['p-value', f"{comparison['sbm']['p']:.2e}",
         f"{comparison['lfr']['p']:.2e}"],
        ['Mean CI', f"±{np.mean(comparison['sbm']['cis']):.3f}",
         f"±{np.mean(comparison['lfr']['cis']):.3f}"],
        ['Max ρ_HB', f"{max(comparison['sbm']['means']):.3f}",
         f"{max(comparison['lfr']['means']):.3f}"],
        ['ρ at h=0', f"{comparison['sbm']['means'][0]:.3f}",
         f"{comparison['lfr']['means'][0]:.3f}"]
    ]

    table = ax5.table(cellText=table_data, cellLoc='center',
                     loc='center', colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(weight='bold')

    # Highlight better values
    # Spearman r (row 1) - higher is better
    if comparison['lfr']['r'] > comparison['sbm']['r']:
        table[(1, 2)].set_facecolor('#C8E6C9')
    else:
        table[(1, 1)].set_facecolor('#C8E6C9')

    # Mean CI (row 3) - lower is better
    if np.mean(comparison['lfr']['cis']) < np.mean(comparison['sbm']['cis']):
        table[(3, 2)].set_facecolor('#C8E6C9')
    else:
        table[(3, 1)].set_facecolor('#C8E6C9')

    ax5.set_title('Panel E: Statistical\nSummary',
                  fontsize=14, fontweight='bold', pad=20)

    plt.suptitle('Hub-Bridging Generator Comparison: HB-SBM vs HB-LFR',
                 fontsize=18, fontweight='bold', y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\nFigure saved to: {save_path}")

        # Also save as PDF for publication
        pdf_path = str(save_path).replace('.png', '.pdf')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"PDF saved to: {pdf_path}")

    return fig


def generate_latex_table(comparison):
    """
    Generate LaTeX table for paper.
    """
    print("\n" + "="*70)
    print("LATEX TABLE FOR PAPER")
    print("="*70)

    latex = r"""
\begin{table}[h]
\centering
\caption{Comparison of HB-SBM and HB-LFR Generators on Parameter Control Validation}
\label{tab:generator_comparison}
\begin{tabular}{lcc}
\hline
\textbf{Metric} & \textbf{HB-SBM} & \textbf{HB-LFR} \\
\hline
"""

    # Determine which values to bold (better performance)
    sbm_r = comparison['sbm']['r']
    lfr_r = comparison['lfr']['r']
    sbm_ci = np.mean(comparison['sbm']['cis'])
    lfr_ci = np.mean(comparison['lfr']['cis'])
    sbm_max = max(comparison['sbm']['means'])
    lfr_max = max(comparison['lfr']['means'])

    # Spearman r - higher is better
    if lfr_r > sbm_r:
        latex += f"Spearman correlation (r) & {sbm_r:.3f} & \\textbf{{{lfr_r:.3f}}} \\\\\n"
    else:
        latex += f"Spearman correlation (r) & \\textbf{{{sbm_r:.3f}}} & {lfr_r:.3f} \\\\\n"

    latex += f"Statistical significance (p) & {comparison['sbm']['p']:.2e} & {comparison['lfr']['p']:.2e} \\\\\n"

    # CI - lower is better
    if lfr_ci < sbm_ci:
        latex += f"Mean 95\\% CI width & ±{sbm_ci:.3f} & \\textbf{{±{lfr_ci:.3f}}} \\\\\n"
    else:
        latex += f"Mean 95\\% CI width & \\textbf{{±{sbm_ci:.3f}}} & ±{lfr_ci:.3f} \\\\\n"

    # Max rho - higher is better
    if lfr_max > sbm_max:
        latex += f"Maximum $\\rho_{{HB}}$ achieved & {sbm_max:.3f} & \\textbf{{{lfr_max:.3f}}} \\\\\n"
    else:
        latex += f"Maximum $\\rho_{{HB}}$ achieved & \\textbf{{{sbm_max:.3f}}} & {lfr_max:.3f} \\\\\n"

    latex += f"Baseline $\\rho_{{HB}}$ (h=0) & {comparison['sbm']['means'][0]:.3f} & {comparison['lfr']['means'][0]:.3f} \\\\\n"

    latex += r"""\hline
\multicolumn{3}{l}{\small \textit{Bold indicates superior performance}} \\
\end{tabular}
\end{table}
"""

    print(latex)

    return latex


def print_detailed_results(comparison):
    """
    Print detailed results table.
    """
    print("\n" + "="*70)
    print("DETAILED RESULTS TABLE")
    print("="*70)

    h_values = comparison['h_values']

    print(f"\n{'h':>6} | {'HB-SBM ρ_HB':>20} | {'HB-LFR ρ_HB':>20}")
    print("-" * 52)

    for i, h in enumerate(h_values):
        sbm_mean = comparison['sbm']['means'][i]
        sbm_ci = comparison['sbm']['cis'][i]
        lfr_mean = comparison['lfr']['means'][i]
        lfr_ci = comparison['lfr']['cis'][i]

        print(f"{h:>6.2f} | {sbm_mean:>8.3f} ± {sbm_ci:<8.3f} | {lfr_mean:>8.3f} ± {lfr_ci:<8.3f}")

    print("-" * 52)
    print(f"\n{'Metric':<25} | {'HB-SBM':>12} | {'HB-LFR':>12}")
    print("-" * 55)
    print(f"{'Spearman correlation':<25} | {comparison['sbm']['r']:>12.3f} | {comparison['lfr']['r']:>12.3f}")
    print(f"{'p-value':<25} | {comparison['sbm']['p']:>12.2e} | {comparison['lfr']['p']:>12.2e}")
    print(f"{'Mean 95% CI':<25} | {np.mean(comparison['sbm']['cis']):>12.3f} | {np.mean(comparison['lfr']['cis']):>12.3f}")
    print(f"{'Max ρ_HB':<25} | {max(comparison['sbm']['means']):>12.3f} | {max(comparison['lfr']['means']):>12.3f}")
    print(f"{'Range (max-min)':<25} | {max(comparison['sbm']['means'])-min(comparison['sbm']['means']):>12.3f} | {max(comparison['lfr']['means'])-min(comparison['lfr']['means']):>12.3f}")


if __name__ == '__main__':
    print("="*70)
    print("COMPARATIVE ANALYSIS: HB-SBM vs HB-LFR")
    print("="*70)

    # Run comparison (loads existing or generates new data)
    comparison = compare_generators()

    # Print detailed results
    print_detailed_results(comparison)

    # Generate comprehensive figure
    results_dir = Path(__file__).parent.parent / 'data' / 'results'
    fig = plot_comprehensive_comparison(
        comparison,
        save_path=str(results_dir / 'figure_generator_comparison.png')
    )

    # Generate LaTeX table
    latex_table = generate_latex_table(comparison)

    # Save comparison data
    comparison_path = results_dir / 'generator_comparison.pkl'
    with open(comparison_path, 'wb') as f:
        pickle.dump(comparison, f)
    print(f"\nComparison data saved to: {comparison_path}")

    # Save LaTeX table to file
    latex_path = results_dir / 'table_generator_comparison.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {latex_path}")

    print("\n" + "="*70)
    print("✓ Comparative analysis complete!")
    print("="*70)
