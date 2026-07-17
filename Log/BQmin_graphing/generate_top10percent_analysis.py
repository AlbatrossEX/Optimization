"""
Top 10% Analysis: Generate visualization of best-case performance

This script analyzes the top 10% best-performing trials from 5000 comparative
trials between BQMIN and interpolation minimum strategies, generating a
four-panel visualization showing:
1. Radius distribution in top 10% vs all trials
2. Winner distribution comparison
3. Improvement analysis scatter plot
4. Head-to-head advantage sorted view

Run: python3 Log/BQmin_graphing/generate_top10percent_analysis.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # Log/BQmin_graphing/

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bq_common import GRAPH_DIR, build_study_problem, compare_once

GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
N_TRIALS = 5000
DIM = 3
RADIUS_RANGE = (0.01, 5.0)
N_BINS = 6
X_RANGE = (-2.0, 2.0)

# Initialize problem
Function_object = build_study_problem(m=15, nprob=8)


def main():
    """Run comparative analysis and generate visualization."""

    # Generate trial data
    print("Running 5000 trials...")
    rng = np.random.default_rng(42)
    radii = np.exp(rng.uniform(np.log(RADIUS_RANGE[0]), np.log(RADIUS_RANGE[1]), N_TRIALS))
    f_origin = np.empty(N_TRIALS)
    f_interp = np.empty(N_TRIALS)
    f_bq = np.empty(N_TRIALS)

    for trial in range(N_TRIALS):
        x = rng.uniform(*X_RANGE, DIM)
        f_origin[trial], f_interp[trial], f_bq[trial] = compare_once(
            Function_object, x, radii[trial], DIM
        )
        if (trial + 1) % 1000 == 0:
            print(f"  {trial + 1}/5000")

    Function_object.flush_log()

    # Identify top 10%
    all_best = np.minimum(f_interp, f_bq)
    top10_threshold = np.percentile(all_best, 10)
    top10_mask = all_best <= top10_threshold
    top10_count = np.sum(top10_mask)

    print(f"\n=== TOP 10% ANALYSIS ===")
    print(f"Top 10% threshold: {top10_threshold:.6f}")
    print(f"Number of top-10% trials: {top10_count}")

    # Extract top 10% data
    top10_radii = radii[top10_mask]
    top10_f_origin = f_origin[top10_mask]
    top10_f_interp = f_interp[top10_mask]
    top10_f_bq = f_bq[top10_mask]

    # Analyze winners
    stacked = np.vstack([top10_f_origin, top10_f_interp, top10_f_bq])
    winner_top10 = np.argmin(stacked, axis=0)

    stacked_all = np.vstack([f_origin, f_interp, f_bq])
    winner_all = np.argmin(stacked_all, axis=0)

    print(f"\nWinner distribution in TOP 10%:")
    for k, name in enumerate(["origin", "interp", "bqmin"]):
        n = int(np.sum(winner_top10 == k))
        pct = 100 * n / top10_count
        print(f"  {name:15}: {n:4d} ({pct:5.1f}%)")

    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Radius distribution comparison
    ax1.hist(radii[~top10_mask], bins=50, alpha=0.5, label='All other trials (90%)',
             color='gray', edgecolor='black')
    ax1.hist(top10_radii, bins=50, alpha=0.7, label='Top 10% best',
             color='red', edgecolor='darkred')
    ax1.set_xscale('log')
    ax1.set_xlabel('Trust-region radius', fontsize=11)
    ax1.set_ylabel('Number of trials', fontsize=11)
    ax1.set_title('Radius Distribution: Top 10% vs All Trials', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Panel 2: Winner distribution comparison
    methods = ['origin', 'interp', 'bqmin']
    top10_wins = [int(np.sum(winner_top10 == k)) for k in range(3)]
    all_wins = [int(np.sum(winner_all == k)) for k in range(3)]

    x_pos = np.arange(len(methods))
    width = 0.35
    bars1 = ax2.bar(x_pos - width/2, [100*w/top10_count for w in top10_wins], width,
                    label='Top 10%', color='red', alpha=0.7, edgecolor='darkred')
    bars2 = ax2.bar(x_pos + width/2, [100*w/N_TRIALS for w in all_wins], width,
                    label='All trials', color='gray', alpha=0.5, edgecolor='black')

    ax2.set_ylabel('Win percentage (%)', fontsize=11)
    ax2.set_title('Winner Distribution: Top 10% vs Overall', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods, fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    # Annotate bars
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=9)

    # Panel 3: Improvement scatter
    top10_imp_interp = top10_f_origin - top10_f_interp
    top10_imp_bq = top10_f_origin - top10_f_bq

    ax3.scatter(top10_imp_interp, top10_imp_bq, alpha=0.5, s=20,
                color='purple', edgecolors='none')

    # Reference line
    min_val = min(top10_imp_interp.min(), top10_imp_bq.min())
    max_val = max(top10_imp_interp.max(), top10_imp_bq.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1,
             label='Equal improvement')
    ax3.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax3.axvline(0, color='gray', linestyle=':', alpha=0.5)

    ax3.set_xlabel('Improvement (interpolation over origin)', fontsize=11)
    ax3.set_ylabel('Improvement (BQMIN over origin)', fontsize=11)
    ax3.set_title('Improvement Comparison in Top 10%\n(points above diagonal = BQMIN better)',
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)

    # Panel 4: Head-to-head sorted advantage
    advantage = top10_f_interp - top10_f_bq
    advantage_sorted = np.sort(advantage)

    ax4.plot(np.arange(len(advantage_sorted)), advantage_sorted, 'o-', color='purple',
             markersize=3, linewidth=1, label='f_interp - f_bqmin')
    ax4.axhline(0, color='k', linestyle='--', linewidth=1, label='Equal')
    ax4.fill_between(np.arange(len(advantage_sorted)), 0, advantage_sorted,
                     where=(advantage_sorted > 0), alpha=0.3, color='orange',
                     label='BQMIN better')
    ax4.fill_between(np.arange(len(advantage_sorted)), 0, advantage_sorted,
                     where=(advantage_sorted <= 0), alpha=0.3, color='blue',
                     label='Interp better')

    ax4.set_xlabel('Trial rank (sorted by advantage)', fontsize=11)
    ax4.set_ylabel('f_interp - f_bqmin\n(positive = BQMIN better)', fontsize=11)
    ax4.set_title(f'Head-to-Head Comparison in Top 10% ({top10_count} trials)',
                  fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3)

    # Main title
    fig.suptitle(
        f'Top 10% Analysis: {top10_count} Best-Performing Trials\n(Best objective values: ≤ {top10_threshold:.6f})',
        fontsize=14, fontweight='bold'
    )
    fig.tight_layout()

    # Save figure
    out_path = GRAPH_DIR / "BQmin_TOP10PERCENT.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nVisualization saved to {out_path}")

    # Print summary statistics
    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Top 10% radius range: {top10_radii.min():.6f} to {top10_radii.max():.6f}")
    print(f"Top 10% median radius: {np.median(top10_radii):.6f}")
    print(f"Top 10% mean radius: {top10_radii.mean():.6f}")

    print(f"\nImprovement analysis (TOP 10%):")
    print(f"  Interpolation mean improvement: {top10_imp_interp.mean():+.6f}")
    print(f"  BQMIN mean improvement:         {top10_imp_bq.mean():+.6f}")
    print(f"  BQMIN worst case:               {top10_imp_bq.min():+.6f}")

    print(f"\nHead-to-head in TOP 10%:")
    print(f"  BQMIN better: {int(np.sum(advantage > 0))} trials")
    print(f"  Interp better: {int(np.sum(advantage < 0))} trials")
    print(f"  Median advantage: {np.median(advantage):+.6f}")


if __name__ == "__main__":
    main()
