"""Run machinery for trust-region experiments.

Nothing here defines *what* is run: the function, constants, starting points,
radii, gh types, methods, evaluation budget and case distribution are all
declared by an entry point in Running/ and arrive as an Experiment. This module
owns *how* an experiment is executed — process-pool fan-out, log routing and
progress reporting — so adding a new set of experiments never means editing it.

The helpers here (build_cases, log_radii, effort_radii, random_starts) are
generators, not configuration: every value they produce comes from their
caller's arguments.

Logs: organized by evaluation budget first, then by experiment. A run writes
into Log/Logs/<evaluations> Evalu/<Name>/, found or created on the way in, so
the same execution code run at the same budget always lands in the same
directory (same-named logs from an earlier run are replaced).

Resume: a suite that is cut short (closed terminal, sleep, crash) continues
from where it left off when relaunched — cases whose final log already exists
are skipped (see pending_cases). Run an entry script with --fresh to redo
every case instead.
"""
import os
import re
import sys
import numpy as np
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from numpy.typing import NDArray
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from construct_functions import build_problem
from general_model.Trust_region_optimization import TR_function

Array1D = NDArray[np.floating]
# (miu, theta, shrink, extend, radius, p, method, gh_type) — declared by the
# entry point in Running/ and passed in; nothing here hardcodes run parameters.
Constants = Tuple[float, float, float, float, float, float, int, int]
Case = Tuple[str, Array1D, float, int, int]  # (label, x_0, radius, method, gh_type)

# Root under which every run's logs live. A run of an experiment writes into
# the budget-keyed subdirectory <evaluations> Evalu/<Name>/ below this; this
# folder itself is never wiped.
# Anchored to the project root so runs work regardless of the caller's cwd.
LOG_ROOT = Path(__file__).resolve().parents[1] / "Log" / "Logs"
# How many pipelines (worker processes) the suite fans out into. The objective is
# CPU-bound, so throughput peaks at one worker per logical core: benchmarking this
# machine showed 16 workers (= os.cpu_count()) beating both 8 and an oversubscribed
# 24/32 (the latter ~20% SLOWER, since extra processes only time-slice the same
# cores and add context-switching). So the suite uses every core and no more.
# Set explicitly rather than left None so full-core use does not depend on the
# Python/OS default; override per call with run_experiment(exp, max_workers=...).
# The other speed lever — pricing a poised set across threads inside one worker —
# is disabled here for the same reason (see TR_function._eval_workers) and left
# for single-process callers with idle cores.
MAX_WORKERS: Optional[int] = os.cpu_count()

METHOD_NAMES = {
    0: "bqmin step",
    1: "best interpolation point",
    2: "better of the two",
    3: "dynamic interpolation point",
}


@dataclass
class Experiment:
    """A complete description of what to run, assembled by a Running/ entry point.

    name        names this experiment; a run writes its logs into the shared
                Log/Logs/<evaluations> Evalu/<Name>/ directory (found or
                created; re-runs at the same budget replace same-named logs).
    problem     is a picklable build_problem spec; workers each build their own.
    evaluations is the function-evaluation budget per run (the stopping condition).
    prefix      is the label prefix stamped on this experiment's log filenames.
    """

    name: str
    problem: Dict
    constants: Constants
    cases: List[Case]
    evaluations: int
    prefix: str


# --- Case construction (generic; all values come from the caller) -------------


def log_radii(low: float, high: float, count: int) -> List[float]:
    """count log-spaced radii in [low, high]. Rounded to 3 significant figures,
    which keeps the radius readable in the log filename without collapsing the span."""
    return [float(f"{r:.3g}") for r in np.logspace(np.log10(low), np.log10(high), count)]


def effort_radii(
    count: int,
    low: float = 0.01,
    mid: float = 1.0,
    high: float = 3.0,
    low_frac: float = 0.9,
) -> List[float]:
    """`count` radii with most of the sampling effort in [low, mid] and the rest
    in (mid, high]: by default 90% of the radii fall in [0.01, 1] and 10% in
    (1, 3]. The lower band is log-spaced (it spans two orders of magnitude); the
    narrow upper band is linearly spaced. Rounded to 3 significant figures.

    This concentrates the comparison where the interesting trust-region
    behaviour lives (radius below 1) while still probing a few larger radii.
    """
    n_low = max(1, round(count * low_frac))
    n_high = max(1, count - n_low)
    lows = np.logspace(np.log10(low), np.log10(mid), n_low)
    # exclude the shared endpoint `mid` so it is not sampled twice
    highs = np.linspace(mid, high, n_high + 1)[1:]
    radii = sorted(float(f"{r:.3g}") for r in np.concatenate([lows, highs]))
    return radii


def random_starts(
    count: int, dim: int, box: float, seed: int, problem: Optional[Dict] = None
) -> Array1D:
    """count starting points drawn uniformly from [-box, box]^dim, rounded to 2dp.

    problem: optional build_problem spec (the experiment's PROBLEM dict). When
    given, any draw where the objective is not finite is rejected and redrawn,
    so no run starts where the function is infinite — e.g. the nondiff Bard
    function (nonsmooth nprob 8) is inf on the whole quadrant x[1] <= 0,
    x[2] <= 0 because calfun clamps x to max(x, 0) there, zeroing Bard's
    denominator. A start inside such a region gives the solver nothing to
    rank, so the run measures escape behaviour instead of the method.

    Draws come one point at a time off the same generator stream that the
    unfiltered call consumes row by row, so accepted points match the
    problem=None starts until the first rejection, and everything stays
    deterministic per seed.
    """
    rng = np.random.default_rng(seed)
    if problem is None:
        return np.round(rng.uniform(-box, box, size=(count, dim)), 2)
    # .f, not .output(): probing must not count evaluations or write log lines.
    f = build_problem(**problem).f
    starts: List[Array1D] = []
    # A rejected draw IS a division by zero inside the objective; the probe
    # exists to catch those, so numpy need not also warn about them.
    with np.errstate(divide="ignore", invalid="ignore"):
        while len(starts) < count:
            x = np.round(rng.uniform(-box, box, size=dim), 2)
            if np.isfinite(f(x)):
                starts.append(x)
    return np.array(starts)


def build_cases(
    starts: Sequence[Array1D],
    radii: Sequence[float],
    combos: Sequence[Tuple[int, int]],
    prefix: str,
) -> List[Case]:
    """The full starts x radii x combos grid, where combos are (method, gh_type) pairs.

    One uniform grid. For a distribution that varies per slice — different radii
    or starts for different method/gh pairs — declare Blocks and use build_blocks.
    """
    cases: List[Case] = []
    for x_0 in starts:
        for radius in radii:
            for method, gh_type in combos:
                cases.append(
                    (
                        f"{prefix}{len(cases):03d}",
                        np.asarray(x_0, dtype=float),
                        float(radius),
                        int(method),
                        int(gh_type),
                    )
                )
    return cases


@dataclass
class Block:
    """One slice of a case distribution: every combination of its starts, radii,
    methods and gh_types.

    methods x gh_types is a cross product, which is what makes "hold the
    trust-region method fixed, vary the model builder" a one-liner:
    Block(methods=[0], gh_types=[0, 1], ...). Each block carries its own starts
    and radii, so one comparison can sweep a denser radius grid than another.
    """

    methods: Sequence[int]
    gh_types: Sequence[int]
    starts: Sequence[Array1D]
    radii: Sequence[float]


def build_blocks(blocks: Sequence[Block], prefix: str) -> List[Case]:
    """Expand blocks and concatenate them into one case set.

    Labels run sequentially across the whole set (prefix000, prefix001, ...), so
    no two cases can collide on a log filename however the blocks overlap.
    Overlapping blocks are allowed and meaningful: gh_type 1 draws a random
    model, so repeating an identical case is how you average over that draw.
    """
    cases: List[Case] = []
    for block in blocks:
        for x_0 in block.starts:
            for radius in block.radii:
                for method in block.methods:
                    for gh_type in block.gh_types:
                        cases.append(
                            (
                                f"{prefix}{len(cases):03d}",
                                np.asarray(x_0, dtype=float),
                                float(radius),
                                int(method),
                                int(gh_type),
                            )
                        )
    return cases


_TRAILING_DIGITS = re.compile(r"^(.*?)(\d+)$")


def _label_ranges(labels: Sequence[str]) -> str:
    """Compress labels into contiguous runs: N000..N019, N200..N219.

    Blocks interleave, so a start's labels are no longer one unbroken range; a
    plain first..last would overstate what ran.
    """
    parsed = []
    for label in labels:
        match = _TRAILING_DIGITS.match(label)
        if not match:  # labels need not be <stem><number>; fall back to listing
            return ", ".join(labels)
        parsed.append((match.group(1), int(match.group(2)), label))
    parsed.sort(key=lambda item: (item[0], item[1]))

    runs: List[List] = []  # [stem, last_index, first_label, last_label]
    for stem, index, label in parsed:
        if runs and runs[-1][0] == stem and index == runs[-1][1] + 1:
            runs[-1][1], runs[-1][3] = index, label
        else:
            runs.append([stem, index, label, label])
    return ", ".join(
        first if first == last else f"{first}..{last}" for _, _, first, last in runs
    )


def describe_cases(cases: List[Case], evaluations: int) -> str:
    """Human-readable details and distribution of a case set: which radii,
    methods, gh types, and starting points run, and the evaluation budget each gets."""
    radii = Counter(float(c[2]) for c in cases)
    methods = Counter(int(c[3]) for c in cases)
    gh_types = Counter(int(c[4]) for c in cases)
    combos = Counter((int(c[3]), int(c[4])) for c in cases)
    starts: Dict[Tuple[float, ...], List[str]] = {}  # start -> labels running it
    for label, x_0, *_ in cases:
        starts.setdefault(tuple(np.asarray(x_0, dtype=float).tolist()), []).append(label)

    # radii swept by each (method, gh_type), so a comparison that holds one fixed
    # and varies the other is readable straight off the summary
    combo_radii: Dict[Tuple[int, int], set] = {}
    for _, _, radius, method, gh_type in cases:
        combo_radii.setdefault((int(method), int(gh_type)), set()).add(float(radius))

    header = f"case set: {len(cases)} runs ({evaluations} function evaluations each)"
    # Only claim a cross product when the numbers actually multiply out; blocks
    # with per-block radii/starts generally do not.
    if len(starts) * len(radii) * len(combos) == len(cases):
        header += (
            f" = {len(starts)} starting points x {len(radii)} radii "
            f"x {len(combos)} method/gh combos"
        )
    else:
        header += (
            f": {len(starts)} starting points, {len(radii)} distinct radii, "
            f"{len(combos)} method/gh combos"
        )

    lines = [
        header,
        "  radii: " + ", ".join(f"{r:g} ({n} runs)" for r, n in sorted(radii.items())),
        "  methods: " + ", ".join(
            f"{m} = {METHOD_NAMES.get(m, 'unknown')} ({n} runs)" for m, n in sorted(methods.items())
        ),
        "  gh types: " + ", ".join(f"{g} ({n} runs)" for g, n in sorted(gh_types.items())),
        "  method/gh combos:",
    ]
    for (method, gh_type), n in sorted(combos.items()):
        swept = sorted(combo_radii[(method, gh_type)])
        span = f"{len(swept)} radii {min(swept):g}..{max(swept):g}"
        lines.append(
            f"    method {method} ({METHOD_NAMES.get(method, 'unknown')}) x gh {gh_type}: "
            f"{n} runs, {span}"
        )
    lines.append("  starting points:")
    for start, labels in starts.items():
        point = "[" + ", ".join(f"{v:g}" for v in start) + "]"
        lines.append(f"    {_label_ranges(labels)}  {point}  ({len(labels)} runs)")
    return "\n".join(lines)


# --- Logging -----------------------------------------------------------------


def _dir_name(name: str) -> str:
    """Directory form of an experiment name: first letter capitalized, matching
    the Log/Logs/<N> Evalu/<Name>/ template (e.g. Smooth_four_methods)."""
    return name[:1].upper() + name[1:]


def run_dir_for(name: str, evaluations: int, root: Path = LOG_ROOT) -> Path:
    """Find or create the log directory for one experiment at one budget:
    Log/Logs/<evaluations> Evalu/<Name>/.

    Logs are organized by evaluation budget first, then by experiment, so the
    same execution code run at the same budget always writes into the same
    directory; same-named logs from an earlier run are replaced.
    """
    run_dir = root / f"{int(evaluations)} Evalu" / _dir_name(name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def latest_run_dir(name: str, root: Path = LOG_ROOT) -> Optional[Path]:
    """The most recently written log directory for an experiment, or None.

    Searches the budget-keyed layout Log/Logs/<N> Evalu/<Name>/ (when several
    budgets exist, the most recently modified wins) and falls back to any
    legacy timestamped Log/Logs/<name>_<timestamp>/ directories. Graph scripts
    use this to find the logs from the last run without being told the budget."""
    candidates = [p for p in root.glob(f"* Evalu/{_dir_name(name)}") if p.is_dir()]
    candidates += [p for p in root.glob(f"{name}_*") if p.is_dir()]
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def log_name(label: str, x_0: Array1D, radius: float, p: float, method: int, gh_type: int) -> str:
    start = "-".join(f"{v:g}" for v in np.asarray(x_0, dtype=float))
    scenario = f"{label}_" if label else ""
    return f"{scenario}start{start}_radius{radius:g}_p{p:g}_method{method}_gh{gh_type}"


# --- Single run --------------------------------------------------------------


def main(
    problem: TR_function,
    constants: Constants,
    x_0: Array1D,
    evaluations: int,
    radius: float = None,
    method: int = None,
    gh_type: int = None,
    label: str = "",
    name: str = "single",
) -> Array1D:
    """Single optimization run on an already-constructed problem object.

    Writes one log into the shared Log/Logs/<evaluations> Evalu/<Name>/
    directory (found or created); a same-named log from an earlier run at the
    same budget is replaced.
    """
    miu, theta, shrink, extend, radius_c, p, method_c, gh_type_c = constants
    if radius is None:
        radius = radius_c
    if method is None:
        method = method_c
    if gh_type is None:
        gh_type = gh_type_c

    run_dir = run_dir_for(name, evaluations)
    live = run_dir / "New.txt"
    problem.redirect_log(str(live))

    result = problem.trust_region_optimization_function(
        method=method,
        x_0=x_0,
        miu=miu,
        theta=theta,
        shrink=shrink,
        extend=extend,
        radius=radius,
        p=p,
        max_evals=evaluations,
        gh_type=gh_type,
    )
    print(result)
    print(problem.output(result))
    problem.flush_log()
    name_txt = log_name(label, x_0, radius, p, method, gh_type)
    # replace(), not rename(): re-running the same case at the same budget
    # overwrites the previous log (rename would raise on Windows).
    live.replace(run_dir / f"{name_txt}.txt")
    return result


# --- Parallel suite ----------------------------------------------------------
# Every case gets its own log file inside the run's directory; the per-worker
# live log New_<pid>.txt keeps parallel writers from colliding.

# Per-worker state, set once by _init_worker: the problem object holds lambdas
# and a logger thread, so it cannot be pickled — each worker builds its own
# from the spec instead of receiving the parent's.
_PROBLEM: TR_function = None
_CONSTANTS: Constants = None
_RUN_DIR: Path = None


def _init_worker(problem_spec: Dict, constants: Constants, run_dir: str) -> None:
    global _PROBLEM, _CONSTANTS, _RUN_DIR
    _RUN_DIR = Path(run_dir)
    _RUN_DIR.mkdir(parents=True, exist_ok=True)
    _PROBLEM = build_problem(**problem_spec)
    # Parallelism here is across processes (one worker per pipeline), so each
    # worker prices its poised sets serially — per-worker threads would just
    # oversubscribe the cores the pool is already saturating.
    _PROBLEM._eval_workers = 1
    _CONSTANTS = constants
    # One live log per worker process, inside this run's directory; renamed to the
    # final name after each case.
    _PROBLEM.redirect_log(str(_RUN_DIR / f"New_{os.getpid()}.txt"))


def run_case(case: Case, evaluations: int) -> Tuple[str, str, float]:
    label, x_0, radius, method, gh_type = case
    miu, theta, shrink, extend, _, p, _, _ = _CONSTANTS

    # A previous case that crashed in this worker leaves a stale live log
    # behind; drop it so its lines don't leak into this case's log.
    live = _RUN_DIR / f"New_{os.getpid()}.txt"
    live.unlink(missing_ok=True)

    result = _PROBLEM.trust_region_optimization_function(
        method=method,
        x_0=x_0,
        miu=miu,
        theta=theta,
        shrink=shrink,
        extend=extend,
        radius=radius,
        p=p,
        max_evals=evaluations,
        gh_type=gh_type,
    )
    f_final = float(_PROBLEM.output(result))
    _PROBLEM.flush_log()

    name = log_name(label, x_0, radius, p, method, gh_type)
    # replace(), not rename(): re-running the same case at the same budget
    # overwrites the previous log (rename would raise on Windows).
    live.replace(_RUN_DIR / f"{name}.txt")
    return label, name, f_final


def run_suite(
    problem_spec: Dict,
    constants: Constants,
    cases: List[Case],
    evaluations: int,
    run_dir: Path,
    max_workers: Optional[int] = MAX_WORKERS,
) -> None:
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(problem_spec, constants, str(run_dir)),
    ) as pool:
        futures = {pool.submit(run_case, case, evaluations): case[0] for case in cases}
        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                label, name, f_final = future.result()
            except Exception as exc:  # one bad case should not kill the suite
                print(f"[{done}/{len(cases)}] {futures[future]}: FAILED ({exc})")
                continue
            print(f"[{done}/{len(cases)}] {label}: f = {f_final:.6g}  ({name})")


def pending_cases(experiment: Experiment, run_dir: Path) -> List[Case]:
    """The cases whose final log does not exist in run_dir yet.

    A case streams its output to a per-worker live log (New_<pid>.txt) and only
    gains its final-named log when it ran to completion (run_case renames it as
    its last act), so a case with a final log is always a finished one and an
    interrupted case is always still pending. This is what makes an interrupted
    suite resumable: relaunching runs exactly the cases that did not finish.
    """
    p = experiment.constants[5]
    return [
        case
        for case in experiment.cases
        if not (run_dir / f"{log_name(case[0], case[1], case[2], p, case[3], case[4])}.txt").exists()
    ]


def run_experiment(
    experiment: Experiment,
    max_workers: Optional[int] = MAX_WORKERS,
    fresh: Optional[bool] = None,
) -> Path:
    """Report, then run every case on the pool, writing all logs into the
    experiment's budget-keyed directory (found or created). Returns that
    directory. Re-runs at the same budget land in the same place.

    Resumable: cases that already have a final log in the run directory are
    skipped, so a suite cut short (closed terminal, sleep, crash) continues
    from where it left off when simply relaunched. Only completed cases have
    final logs (see pending_cases), so nothing half-written is ever kept. Pass
    fresh=True — or run the entry script with --fresh — to redo every case;
    by default fresh follows the command line."""
    if fresh is None:
        fresh = "--fresh" in sys.argv
    run_dir = run_dir_for(experiment.name, experiment.evaluations)

    # Live logs left by an interrupted run's workers are partial output for
    # cases that will be re-run; drop them so they never pollute the directory.
    for stale in run_dir.glob("New_*.txt"):
        stale.unlink()

    # Details and distribution of what is about to run: printed up front and
    # saved next to this run's logs so it can be inspected after the run too.
    # Always the full case set — the file describes the experiment, not the
    # portion of it this relaunch still has to run.
    summary = describe_cases(experiment.cases, experiment.evaluations)
    print(summary)
    print(f"logs -> {run_dir}")
    (run_dir / "case_distribution.txt").write_text(summary + "\n")

    cases = experiment.cases if fresh else pending_cases(experiment, run_dir)
    finished = len(experiment.cases) - len(cases)
    if finished:
        print(
            f"resuming: {finished}/{len(experiment.cases)} cases already have "
            f"logs, {len(cases)} left to run (--fresh redoes everything)"
        )
    if not cases:
        print("nothing to run: every case already has a log")
        return run_dir

    run_suite(
        experiment.problem,
        experiment.constants,
        cases,
        experiment.evaluations,
        run_dir,
        max_workers,
    )
    return run_dir
