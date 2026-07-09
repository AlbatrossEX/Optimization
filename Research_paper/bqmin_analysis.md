# Why bqmin's advantage over interpolation points collapses near radius 1

Analysis of the sharp performance crossover observed in `Bqmin_compare.py`
(2000 trials, Bard problem, radii log-uniform on [0.01, 5], starting points
uniform in [-2, 2]^3): the bqmin step beats the best interpolation point
~77% of the time at radius 0.01, but only ~12% at radius ~3-5, with the
median advantage flipping sign abruptly between radius ~1 and ~1.4.

Experiments: [radius_analysis.py](radius_analysis.py) and
[frame_check.py](frame_check.py) in this folder. Run from the project root:

```
python3 Research_paper/radius_analysis.py
python3 Research_paper/frame_check.py
```

## Setup

Per trial, at center `x` with trust-region radius `Δ`:

1. `Function_object.GH(x, Δ)` builds a 10-point poised interpolation set via
   `algorithm_6_4` and fits a quadratic model `(G, H)` via `fitfroquad`.
   The **interpolation candidate** is `min f` over the poised points
   (values already computed during the fit — no extra evaluations).
2. The **bqmin candidate** is the true `f(x + s)` at the step
   `s = bqmin(H, G, -Δ, Δ)`, the model minimizer over the inf-norm box.

The objective is `calfun(x, m=15, nprob=8, probtype='smooth')` — the
**Bard function**:

```
f(x) = Σ_{i=1..15} ( y_i − x₀ − i / (x₁·u_i + x₂·v_i) )²
u_i = 16−i,  v_i = min(i, 16−i)
```

## Two stacked mechanisms

### 1. Generic: evaluation vs extrapolation (any non-quadratic f)

The two candidates fail differently:

- The interpolation candidate is an **actually evaluated** point. It can
  never be worse than the center (the poised set contains `x`), and as `Δ`
  grows the set samples more territory, so it monotonically improves.
- The bqmin candidate is the minimizer of a **quadratic extrapolation**,
  and bqmin pushes it to the trust-region boundary — exactly where model
  error is largest. For smooth `f` that error grows like O(Δ³).

One candidate improves with radius, the other degrades: a crossover must
exist for every `f` that is not globally quadratic. Controls in
`radius_analysis.py` confirm both ends:

- `nprob=2` (linear residuals → `f` exactly quadratic): no crossover; with
  a correctly centered gradient the model is exact at every radius (see
  the bug section below).
- `nprob=12` (Box 3D, smooth, no poles): a gradual degradation, not a
  sharp break.

### 2. Function-specific: Bard's pole planes (why it is sharp and why at ~1)

Bard is a sum of squares of *rational* residuals with **8 distinct pole
planes** `x₁·u + x₂·v = 0` fanning through the origin of the (x₁, x₂)
plane. The sampling box [-2, 2]³ straddles this fan.

`radius_analysis.py` adds a diagnostic per trial: does the inf-norm trust
box around `x` intersect any pole plane (`|x₁u + x₂v| ≤ Δ(u+v)`)?
Results (800 trials):

| radius bin | pole in trust box | bqmin median advantage | bqmin win % |
|---|---|---|---|
| 0.01–0.019 | 9% | +2.7 | 76% |
| 0.12–0.42 | 47% | +16 to +26 | 66% |
| 0.78–1.4 | 85% | +13 | 62% |
| 1.4–2.7 | **100%** | **−41** | 32% |
| 2.7–5 | **100%** | **−98** | 12% |

Conditioned across all radii: **no pole in box → bqmin wins 82.9%
(median +9.5); pole in box → bqmin wins 37.6% (median −50)**.

Near a pole, `f` spans orders of magnitude across the poised set, so the
least-squares quadratic is fit through garbage and its minimizer can land
essentially on the pole — producing the −10⁴…−10⁸ outliers visible in the
comparison graph. The min-of-samples candidate is immune: it simply
discards the bad samples.

The break looks sharp rather than gradual because (a) the pole-containment
fraction saturates to 100% right around Δ ≈ 1 given the box geometry, and
(b) the median flips abruptly once contaminated trials become the
majority. In addition, at Δ ≥ 2 the poised set spans most of the [-2, 2]³
box and starts sampling near-global basins, making `f_interp` very good.

## Answer: is it function-specific?

- The **existence** of the crossover is universal for non-quadratic
  functions — model extrapolation must eventually lose to actual
  evaluations as the radius exceeds the scale on which `f` looks quadratic.
- The **location (~1) and sharpness** are specific to Bard's pole geometry
  combined with the [-2, 2]³ sampling box. A pole-free smooth problem
  degrades gradually; an exactly quadratic problem never crosses over.

Practical consequence: the interpolation-point fallback used by solver
methods 1/2 (`trust_region_optimization_1/2`) is nearly free insurance —
it dominates at large radii and essentially never lands worse than the
current iterate.

## Bug found by the quadratic control: mis-centered gradient

On the exactly quadratic control, bqmin *lost* at every radius — impossible
if the pipeline were consistent. Cause (`frame_check.py`):

`algorithm_6_4` returns the poised set in **absolute coordinates**
(`undo_shift_and_scale` at [algorism6_4.py:54](../Smooth/algorism6_4.py)),
so `fitfroquad` fits `f(y) ≈ c + G·y + ½yᵀHy` **about the origin**. The
returned `G` is therefore the model gradient at `y = 0` — but
`trust_region_optimization` (and `Bqmin_compare.py`) use it as the gradient
at the center `x`. The correct center-frame gradient is `G + H @ x`.

Verification against the analytic `calfun` gradient (median relative error
over 20 random points):

| problem | Δ | raw `G` | `G + H @ x` |
|---|---|---|---|
| nprob=2 (quadratic) | 0.01 / 0.1 / 1.0 | ≈ 1.0 (100% wrong) | ~1e-15 (exact) |
| nprob=8 (Bard) | 0.01 | 3.5 | 0.0014 |
| nprob=8 (Bard) | 1.0 | 12.5 | 2.0 |

Re-running the Bard comparison with the re-centered gradient lifts bqmin's
small-radius win rate from ~76% to **~96%**, while the collapse past
radius ~1 remains — confirming the crossover is genuine and not an
artifact of the bug.

Status: **not yet fixed** in the repo. The one-line fix is to return
`G + H @ x, H` from `SmoothFunction.GH` (and identically in
`NonSmoothFunction.GH_1`). It affects every solver iteration; the solvers
currently survive because the ρ-acceptance test filters bad steps.
