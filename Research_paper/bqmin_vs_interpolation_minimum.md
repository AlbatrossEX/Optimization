# bqmin vs. Interpolation Minimum: A Comparison Report

*Compiled from `LOG.md`, `Bqmin_compare.py`, `Research_paper/bqmin_analysis.md`, and the underlying solver code (`Smooth/models/bqmin.py`, `Smooth/algorism6_4.py`, `Smooth/Quad.py`, `Smooth/Class_Smooth.py`) in the `aptimization` project.*

## What the two candidates are

Every iteration of the trust-region solver needs a candidate step away from the current point `x` at radius `Δ`. The project builds two different candidates from the same underlying machinery and has spent considerable effort comparing them.

**The interpolation minimum** is the best function value already sitting in the poised sample set. `SmoothFunction.GH` calls `algorithm_6_4` to build a well-poised set of points around `x` within the trust region, evaluates the true (expensive, black-box) objective `f` at each of those points, and fits a quadratic model `(G, H)` through them via `fitfroquad` in `Quad.py`. Because the poised set always contains `x` itself, the minimum over these points — `f_interp = min(Function_object.f_poised)` — can never be worse than staying put, and it costs nothing extra since the evaluations already happened during model-fitting.

**The bqmin step** takes the quadratic model `(G, H)` that `fitfroquad` produced and minimizes it over the infinity-norm trust-region box `[x-Δ, x+Δ]` using `Smooth/models/bqmin.py`'s projected-gradient solver: starting from `X = 0`, it repeatedly takes a projected-gradient step `X - t·Projg` (with `t` chosen by an exact line-search along the model's quadratic form, clipped to the box) until the projected gradient is below tolerance or `maxit` is reached. The resulting step `s` is then evaluated at the true objective, `f_bqmin = f(x + s)`. Unlike the interpolation candidate, this point is a **model-based extrapolation** — it is never actually sampled during the fit, so its true objective value has to be evaluated separately.

In short: interpolation-minimum is "the best point I've already measured"; bqmin is "the best point my local quadratic guess thinks exists," which then has to be checked against reality.

## How they were compared

`Bqmin_compare.py` runs a head-to-head empirical study on `calfun(nprob=8, probtype='smooth')` — the Bard function, a sum of squared rational residuals with 8 pole planes through the origin of the (x₁, x₂) plane. Each of 2000 trials draws a random center `x` uniformly in `[-2, 2]³` and a radius log-uniformly on `[0.01, 5]`, computes both candidates via `compare_once()`, and (in the latest version of the log) also tracks a third baseline, `f_origin = f(x)`, to see whether either candidate actually improves on the starting point at all.

Results are cut into radius bins (6 coarse bins for boxplots/win counts, 25 finer bins for a continuous median-advantage curve) and plotted in `BQmin_compare.png` as a four-panel figure: a scatter of the two minima colored by radius, a boxplot of `f_interp − f_bqmin` per radius bin, a boxplot of each candidate's improvement over the origin, and the continuous median-advantage curve with IQR band.

## Head-to-head results

The overall win rate across all trials (most recent run, 2000 trials): **origin 1.4%, interpolation 39.5%, bqmin 59.0%**. But the aggregate number hides a sharp radius dependence that is the real finding of the study:

| radius bin | pole in trust box | bqmin median advantage (f_interp − f_bqmin) | bqmin win % |
|---|---|---|---|
| 0.01–0.019 | 9% | +2.7 | 76% |
| 0.12–0.42 | 47% | +16 to +26 | 66% |
| 0.78–1.4 | 85% | +13 | 62% |
| 1.4–2.7 | 100% | −41 | 32% |
| 2.7–5 | 100% | −98 | 12% |

At small radii bqmin's quadratic model is trustworthy and its extrapolated minimum reliably beats anything in the sample set. Past roughly radius 1 this flips hard: bqmin's median advantage swings from double-digit positive to double-digit (then triple-digit) negative, and its win rate collapses from ~76% down to ~12%. The crossover is abrupt rather than gradual — it happens over roughly one bin, not a slow decline.

## Why the crossover happens: two stacked mechanisms

**1. Generic — evaluation vs. extrapolation.** This part is true for any non-quadratic objective, not just Bard. The interpolation candidate is a real, measured point, and it can only get better as the radius grows and the sample set covers more territory. The bqmin candidate is the minimizer of a quadratic *model*, and `bqmin()` pushes that minimizer all the way to the trust-region boundary — precisely where the model's approximation error is largest (it grows roughly like `O(Δ³)` for smooth `f`). One candidate improves with radius, the other degrades with radius, so a crossover must exist for any `f` that isn't globally quadratic. Two controls in `radius_analysis.py` confirm the boundary cases: an exactly quadratic problem (`nprob=2`) shows *no* crossover at all once the gradient bug below is corrected, while a smooth pole-free problem (`nprob=12`, Box 3D) shows a *gradual* decline rather than a sharp break.

**2. Function-specific — Bard's pole geometry is why the break is sharp and located near radius ≈ 1.** Bard's residuals are rational functions with 8 distinct pole planes fanning through the origin, and the `[-2, 2]³` sampling box straddles that fan. `radius_analysis.py` checks, per trial, whether the inf-norm trust box intersects any pole plane. The fraction of trust regions containing a pole climbs from 9% at radius 0.01 to a full 100% by radius ≈ 1.4 — tracking the collapse almost exactly. Conditioned on this diagnostic: no pole in the box → bqmin wins 82.9% of the time (median advantage +9.5); pole in the box → bqmin wins only 37.6% (median −50). Near a pole, function values across the poised set span many orders of magnitude, the least-squares quadratic fit is effectively fit through noise, and its unconstrained minimizer can land almost on top of the pole — producing the extreme negative outliers (down to −10⁴...−10⁸) visible in the comparison scatter. The interpolation-minimum candidate is immune to this failure mode by construction: it just discards whichever sampled points happen to be bad.

So: the *existence* of a crossover is universal for non-quadratic functions. The *location* (~radius 1) and *sharpness* of this particular crossover are specific to Bard's pole geometry combined with the chosen sampling box.

## A bug that briefly confounded the comparison

The quadratic control case initially showed bqmin *losing* at every radius, which shouldn't be possible for an exactly-quadratic objective. The cause, found via `frame_check.py`: `algorithm_6_4` returns the poised point set in **absolute coordinates** (see `undo_shift_and_scale` in `algorism6_4.py`), so `fitfroquad` actually fits the model `f(y) ≈ c + G·y + ½yᵀHy` **about the origin**, not about the center `x`. The `G` it returns is the gradient at `y = 0`, but both `trust_region_optimization` and `Bqmin_compare.py` were using it as if it were the gradient at `x`. The corrected, center-frame gradient is `G + H @ x`.

Verified against the analytic `calfun` gradient (median relative error over 20 random points): the raw `G` is essentially 100% wrong on the quadratic problem at every tested radius, while `G + H @ x` matches to about `1e-15`. On Bard, raw `G` has 3.5x and 12.5x relative error at radius 0.01 and 1.0 respectively, versus 0.0014 and 2.0 for the corrected version. Re-running the Bard comparison with the fix lifts bqmin's small-radius win rate from ~76% to ~96%, while the crossover past radius ~1 remains — confirming the crossover itself is a genuine effect of the model, not an artifact of the bug. As of the last log entry this fix is **not yet applied** in the solver code; it is a pending one-line change (`return G + H @ x, H` in `SmoothFunction.GH`, and identically in `NonSmoothFunction.GH_1`) that affects every solver iteration, though the ρ-acceptance test in the solvers currently absorbs most of the damage from bad steps.

## Practical consequence for the solver family

This comparison directly motivated the project's three-way solver split in `general_model/Trust_region_optimization.py`:

- **Method 0** (`trust_region_optimization`) takes the bqmin step only — the original design.
- **Method 1** (`trust_region_optimization_1`) takes the best interpolation point only.
- **Method 2** (`trust_region_optimization_2`) takes whichever of the two candidates is better, applying the standard ρ-acceptance test to the winner.

Head-to-head convergence tests (`Log/Submin_graphing.py`, three scenarios × three methods) matched the crossover story: at small radius (0.1) all three methods converge similarly; at moderate/large radius (1.0, 10.0) method 0 (bqmin-only) stalls well short of the optimum while methods 1 and 2 continue descending, with method 2 matching or beating the other two in every scenario. The interpolation-point fallback costs nothing extra (its evaluations are already paid for during model-fitting) and, per this data, is "nearly free insurance": it dominates at large radii and — because the poised set always contains the current point — essentially never lands worse than not moving at all.

## Summary table

| | Interpolation minimum | bqmin step |
|---|---|---|
| Source | Best `f` already measured in the poised sample set | True `f` at the quadratic model's minimizer over the trust-region box |
| Cost | Free (reuses model-fit evaluations) | One extra function evaluation |
| Worst case | Never worse than the current point (set contains `x`) | Can be far worse than the current point (extrapolation into model error, worst near poles) |
| Small radius (≲ 0.1–0.5) | Reliable but rarely optimal | Wins most of the time (~76–96% depending on gradient-bug fix) |
| Large radius (≳ 1.5, Bard) | Wins most of the time (median advantage grows into the tens/hundreds) | Collapses toward the pole-contaminated model, wins as low as ~12% |
| Failure mode | None observed in this study | Catastrophic near function poles / high curvature regions the quadratic model can't see |
| Role in final solver | Fallback / floor (method 1), and half of the best-of-both method 2 | Primary candidate at small radius, half of method 2 |
