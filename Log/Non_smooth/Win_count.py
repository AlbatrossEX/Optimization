"""How often each method leads, per function evaluation, across the non-smooth cases.

Reuses the per-case winner logic from Wining_compare: for every case (one
starting condition) and every function-evaluation count, whichever of the three
methods holds the lowest best-so-far objective wins that evaluation. Methods
holding an equal value are split by who reached it first; only a simultaneous
reach counts as a tie.

This script counts those wins over all cases and draws them as a proportion
(stacked area) plot, so the share of cases each method leads can be read off at
any evaluation budget. A second panel shows how many cases are still running at
that budget, which is what the proportions are taken over.

Saves win_count.png into the Graphs folder next to this script.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from Wining_compare import (
    CATEGORIES,
    COLORS,
    NAMES,
    case_winners,
    group_cases,
    graph_dir,
    load_gh0_runs,
)


def win_counts(cases):
    """(grid, counts, active) -- counts[c][k] = cases whose evaluation k+1 is won by c.

    A case only votes while it is still running: past its own last evaluation it
    is dropped rather than having its final winner held forever, so the
    proportions always describe cases that are genuinely active at that budget.
    """
    per_case = [case_winners(case_runs) for case_runs in cases.values()]
    per_case = [(g, b, w) for g, b, w in per_case if g.size]
    last = max(int(g[-1]) for g, _, _ in per_case)

    grid = np.arange(1, last + 1)
    counts = {c: np.zeros(last, dtype=int) for c in CATEGORIES}
    active = np.zeros(last, dtype=int)
    for case_grid, _best, winner in per_case:
        # Index by evaluation number: a case's grid does not necessarily start at
        # evaluation 1 (it starts once all three methods are finite).
        lo, hi = int(case_grid[0]) - 1, int(case_grid[-1])
        active[lo:hi] += 1
        for c in CATEGORIES:
            counts[c][lo:hi] += winner == c
    return grid, counts, active


def win_count_figure(cases, outfile="win_count.png"):
    grid, counts, active = win_counts(cases)

    fig, (ax, ax_n) = plt.subplots(
        2, 1, sharex=True, figsize=(11, 7), gridspec_kw=dict(height_ratios=[3, 1])
    )

    # active is 0 at budgets where no case has started its race yet.
    denom = np.maximum(active, 1)
    shares = [np.where(active > 0, counts[c] / denom, 0.0) for c in CATEGORIES]
    ax.stackplot(
        grid,
        *shares,
        colors=[COLORS[c] for c in CATEGORIES],
        labels=[NAMES[c] for c in CATEGORIES],
        alpha=0.9,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of active cases led")
    ax.set_title(
        f"Which method leads, per evaluation budget ({len(cases)} cases, gh 0)\n"
        "proportion of still-running cases whose best-so-far is lowest"
    )
    # Reversed so the legend reads in the same top-to-bottom order as the stack.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="center left", bbox_to_anchor=(1.01, 0.5))

    ax_n.plot(grid, active, color="#648FFF", lw=1.5)
    ax_n.set_ylim(0, active.max() * 1.1)
    ax_n.set_ylabel("Cases\nstill running")
    ax_n.set_xlabel("Function evaluations")
    ax_n.grid(alpha=0.3)

    ax.set_xscale("log")
    ax.set_xlim(1, grid[-1])

    fig.tight_layout()
    fig.savefig(os.path.join(graph_dir, outfile), dpi=150)
    print(f"saved {outfile}")

    total = sum(int(counts[c].sum()) for c in CATEGORIES)
    print(f"\nevaluation-wins summed over all cases and budgets ({total} decided):")
    for c in CATEGORIES:
        won = int(counts[c].sum())
        print(f"  {NAMES[c]:<24}: {won:7d} ({100 * won / total:5.1f}%)")


if __name__ == "__main__":
    cases = group_cases(load_gh0_runs())
    if not cases:
        raise SystemExit("No complete three-method gh 0 cases found.")
    win_count_figure(cases)
    plt.show()
