import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from construct_functions import Function_object
from Smooth.models.bqmin import bqmin

N_TRIALS = 2000
DIM = 3
RADIUS_RANGE = (0.01, 5.0)   # radii drawn log-uniformly on this interval
N_BINS = 6                   # domains for the boxplot panels and winner counts
N_BINS_FINE = 25             # thinner domains for the median-advantage curve (~80 trials each)
X_RANGE = (-2.0, 2.0)


def compare_once(x, radius):
    """Return (f at the original point, best f over interpolation points,
    f at the bqmin step) for one trial."""
    f_origin = float(Function_object.f(x))

    g, h = Function_object.GH(x, radius)
    f_interp = float(np.min(Function_object.f_poised))

    bound = radius * np.ones(DIM)
    step, _ = bqmin(h, g, -bound, bound)
    f_bq = float(Function_object.f(x + np.asarray(step, dtype=float)))
    return f_origin, f_interp, f_bq


def main():
    rng = np.random.default_rng(42)

    radii = np.exp(rng.uniform(np.log(RADIUS_RANGE[0]),
                               np.log(RADIUS_RANGE[1]), N_TRIALS))
    f_origin = np.empty(N_TRIALS)
    f_interp = np.empty(N_TRIALS)
    f_bq = np.empty(N_TRIALS)

    for trial in range(N_TRIALS):
        x = rng.uniform(*X_RANGE, DIM)
        f_origin[trial], f_interp[trial], f_bq[trial] = compare_once(x, radii[trial])
        print(f"trial {trial + 1:4d}  radius={radii[trial]:<9.4g} "
              f"f_origin={f_origin[trial]:.6g}  "
              f"f_interp={f_interp[trial]:.6g}  f_bqmin={f_bq[trial]:.6g}")

    Function_object.flush_log()

    edges = np.geomspace(RADIUS_RANGE[0], RADIUS_RANGE[1], N_BINS + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    bin_idx = np.clip(np.digitize(radii, edges) - 1, 0, N_BINS - 1)

    stacked = np.vstack([f_origin, f_interp, f_bq])
    winner = np.argmin(stacked, axis=0)
    names = ["original point", "interpolation points", "bqmin step"]
    print(f"\nlowest value over {N_TRIALS} trials:")
    for k, name in enumerate(names):
        n = int(np.sum(winner == k))
        print(f"  {name:<21}: {n}/{N_TRIALS} ({100 * n / N_TRIALS:.1f}%)")

    print("\nper radius domain (winner counts origin/interp/bqmin):")
    for b in range(N_BINS):
        mask = bin_idx == b
        counts = [int(np.sum(winner[mask] == k)) for k in range(3)]
        print(f"  [{edges[b]:8.4g}, {edges[b + 1]:8.4g})  "
              f"origin {counts[0]:3d}  interp {counts[1]:3d}  "
              f"bqmin {counts[2]:3d}  of {int(mask.sum())}")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11))

    norm = matplotlib.colors.LogNorm(vmin=RADIUS_RANGE[0], vmax=RADIUS_RANGE[1])
    sc = ax1.scatter(f_interp, f_bq, c=radii, cmap="viridis", norm=norm,
                     s=14, alpha=0.7, edgecolors="none")
    lims = [min(f_interp.min(), f_bq.min()), max(f_interp.max(), f_bq.max())]
    ax1.plot(lims, lims, "k--", linewidth=1, label="equal minima")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("min f over interpolation points")
    ax1.set_ylabel("f at bqmin step")
    ax1.set_title("Interpolation-point minimum vs bqmin minimum\n"
                  "(below the line = bqmin found a lower point)")
    ax1.legend()
    fig.colorbar(sc, ax=ax1, label="trust-region radius")

    bin_labels = [f"{c:.3g}" for c in centers]

    # bqmin improvement over the best interpolation point, per radius domain.
    diff_groups = [f_interp[bin_idx == b] - f_bq[bin_idx == b]
                   for b in range(N_BINS)]
    ax2.axhline(0, color="k", linewidth=1, linestyle="--")
    ax2.boxplot(diff_groups, labels=bin_labels, showfliers=False)
    for i, group in enumerate(diff_groups, start=1):
        jitter = np.random.default_rng(i).uniform(-0.2, 0.2, group.size)
        ax2.scatter(np.full(group.size, i) + jitter, group,
                    s=6, alpha=0.3, color="tab:blue")
    ax2.set_yscale("symlog")
    ax2.set_xlabel("trust-region radius (domain center)")
    ax2.set_ylabel("f_interp - f_bqmin  (positive = bqmin better)")
    ax2.set_title("Improvement of bqmin over best interpolation point")

    # Improvement over the original point, per radius domain, for both candidates.
    imp_interp = [f_origin[bin_idx == b] - f_interp[bin_idx == b]
                  for b in range(N_BINS)]
    imp_bq = [f_origin[bin_idx == b] - f_bq[bin_idx == b] for b in range(N_BINS)]
    pos = np.arange(1, N_BINS + 1)
    width = 0.32
    ax3.axhline(0, color="k", linewidth=1, linestyle="--",
                label="no better than original point")
    b1 = ax3.boxplot(imp_interp, positions=pos - width / 2, widths=width,
                     showfliers=False, patch_artist=True)
    b2 = ax3.boxplot(imp_bq, positions=pos + width / 2, widths=width,
                     showfliers=False, patch_artist=True)
    for patch in b1["boxes"]:
        patch.set_facecolor("tab:blue")
        patch.set_alpha(0.6)
    for patch in b2["boxes"]:
        patch.set_facecolor("tab:orange")
        patch.set_alpha(0.6)
    ax3.set_xticks(pos)
    ax3.set_xticklabels(bin_labels)
    ax3.set_yscale("symlog")
    ax3.set_xlabel("trust-region radius (domain center)")
    ax3.set_ylabel("improvement over f at original point")
    ax3.set_title("Improvement over the original point\n"
                  "(above 0 = candidate beats the starting point)")
    ax3.legend([b1["boxes"][0], b2["boxes"][0], ax3.lines[0]],
               ["best interpolation point", "bqmin step",
                "no better than original point"])

    # Continuous change of the median advantage of bqmin over interpolation
    # points, on thinner domains (~N_TRIALS / N_BINS_FINE trials each).
    fine_edges = np.geomspace(RADIUS_RANGE[0], RADIUS_RANGE[1], N_BINS_FINE + 1)
    fine_centers = np.sqrt(fine_edges[:-1] * fine_edges[1:])
    fine_idx = np.clip(np.digitize(radii, fine_edges) - 1, 0, N_BINS_FINE - 1)
    fine_groups = [f_interp[fine_idx == b] - f_bq[fine_idx == b]
                   for b in range(N_BINS_FINE)]
    medians = np.array([np.median(g) if g.size else np.nan for g in fine_groups])
    q1 = np.array([np.percentile(g, 25) if g.size else np.nan for g in fine_groups])
    q3 = np.array([np.percentile(g, 75) if g.size else np.nan for g in fine_groups])

    ax4.axhline(0, color="k", linewidth=1, linestyle="--")
    ax4.scatter(radii, f_interp - f_bq, s=5, alpha=0.25, color="gray",
                edgecolors="none", label="individual trials")
    ax4.fill_between(fine_centers, q1, q3, alpha=0.3, color="tab:blue",
                     label="interquartile range")
    ax4.plot(fine_centers, medians, "o-", color="tab:blue",
             label="median f_interp - f_bqmin")
    ax4.set_xscale("log")
    ax4.set_yscale("symlog")
    ax4.set_xlabel("trust-region radius")
    ax4.set_ylabel("f_interp - f_bqmin  (positive = bqmin better)")
    ax4.set_title(f"Median advantage of bqmin over interpolation points\n"
                  f"({N_BINS_FINE} domains, ~{N_TRIALS // N_BINS_FINE} trials each)")
    ax4.legend()

    fig.suptitle(f"{N_TRIALS} trials, radius log-uniform on "
                 f"[{RADIUS_RANGE[0]}, {RADIUS_RANGE[1]}] cut into {N_BINS} domains, "
                 f"random starting points in [{X_RANGE[0]}, {X_RANGE[1]}]^{DIM}")
    fig.tight_layout()
    fig.savefig("BQmin_compare.png", dpi=150)
    print("\nSaved graph to BQmin_compare.png")


if __name__ == "__main__":
    main()
