"""Run machinery for trust-region experiments.

Nothing here defines *what* is run: the function, constants, starting points,
radii, gh types, methods, iteration count and case distribution are all declared
by an entry point in Running/ and arrive as an Experiment. This module owns *how*
an experiment is executed — process-pool fan-out, log routing, cleanup and
progress reporting — so adding a new set of experiments never means editing it.

The helpers here (build_cases, log_radii, random_starts) are generators, not
configuration: every value they produce comes from their caller's arguments.
"""
import os
import re
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

# anchored to the project root so runs work regardless of the caller's cwd
LOG_DIR = Path(__file__).resolve().parents[1] / "Log" / "Logs"
KEEP_LAST_RUNS = 5
# How many pipelines the suite splits into. None = one worker per CPU. This is a
# property of the machine running the suite, not of the experiment.
MAX_WORKERS: Optional[int] = None

METHOD_NAMES = {0: "bqmin step", 1: "best interpolation point", 2: "better of the two"}


@dataclass
class Experiment:
    """A complete description of what to run, assembled by a Running/ entry point.

    name    names the case-distribution file written next to Log/Logs.
    problem is a picklable build_problem spec; workers each build their own.
    prefix  is the label prefix owned by this experiment. Suites share Log/Logs,
            so it is also what run_experiment wipes — an experiment clears its
            own old logs and leaves every other suite's alone.
    """

    name: str
    problem: Dict
    constants: Constants
    cases: List[Case]
    iteration: int
    prefix: str


# --- Case construction (generic; all values come from the caller) -------------


def log_radii(low: float, high: float, count: int) -> List[float]:
    """count log-spaced radii in [low, high]. Rounded to 3 significant figures,
    which keeps the radius readable in the log filename without collapsing the span."""
    return [float(f"{r:.3g}") for r in np.logspace(np.log10(low), np.log10(high), count)]


def random_starts(count: int, dim: int, box: float, seed: int) -> Array1D:
    """count starting points drawn uniformly from [-box, box]^dim, rounded to 2dp."""
    rng = np.random.default_rng(seed)
    return np.round(rng.uniform(-box, box, size=(count, dim)), 2)


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


def describe_cases(cases: List[Case], iteration: int) -> str:
    """Human-readable details and distribution of a case set: which radii,
    methods, gh types, and starting points run, and how many runs each gets."""
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

    header = f"case set: {len(cases)} runs ({iteration} iterations each)"
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


def cleanup_logs(log_dir: Path = LOG_DIR, keep_last: int = KEEP_LAST_RUNS) -> None:
    """Drop crashed-run leftovers, empty logs, and all but the most recent runs."""
    log_dir.mkdir(parents=True, exist_ok=True)
    # live/orphan logs: New.txt plus the per-worker New_<pid>.txt files
    for orphan in log_dir.glob("New*.txt"):
        orphan.unlink()

    runs = [p for p in log_dir.glob("*.txt") if not p.name.startswith("New")]
    for run in runs:
        if run.stat().st_size == 0:
            run.unlink()
    runs = [p for p in runs if p.exists()]

    runs.sort(key=lambda p: p.stat().st_mtime)
    for stale in runs[:-keep_last] if keep_last > 0 else runs:
        stale.unlink()


def log_name(label: str, x_0: Array1D, radius: float, p: float, method: int, gh_type: int) -> str:
    start = "-".join(f"{v:g}" for v in np.asarray(x_0, dtype=float))
    scenario = f"{label}_" if label else ""
    return f"{scenario}start{start}_radius{radius:g}_p{p:g}_method{method}_gh{gh_type}"


# --- Single run --------------------------------------------------------------


def main(
    problem: TR_function,
    constants: Constants,
    x_0: Array1D,
    iteration: int,
    radius: float = None,
    method: int = None,
    gh_type: int = None,
    label: str = "",
    keep_last: int = KEEP_LAST_RUNS,
) -> Array1D:
    """Single optimization run on an already-constructed problem object."""
    miu, theta, shrink, extend, radius_c, p, method_c, gh_type_c = constants
    if radius is None:
        radius = radius_c
    if method is None:
        method = method_c
    if gh_type is None:
        gh_type = gh_type_c

    cleanup_logs(keep_last=keep_last)
    result = problem.trust_region_optimization_function(
        method=method,
        x_0=x_0,
        miu=miu,
        theta=theta,
        shrink=shrink,
        extend=extend,
        radius=radius,
        p=p,
        max_iter=iteration,
        gh_type=gh_type,
    )
    print(result)
    print(problem.output(result))
    problem.flush_log()
    name = log_name(label, x_0, radius, p, method, gh_type)
    (LOG_DIR / "New.txt").rename(LOG_DIR / f"{name}.txt")
    cleanup_logs(keep_last=keep_last)
    return result


# --- Parallel suite ----------------------------------------------------------
# Every case gets its own log file; the per-worker live log New_<pid>.txt keeps
# parallel writers from colliding on a shared New.txt.

# Per-worker state, set once by _init_worker: the problem object holds lambdas
# and a logger thread, so it cannot be pickled — each worker builds its own
# from the spec instead of receiving the parent's.
_PROBLEM: TR_function = None
_CONSTANTS: Constants = None


def _init_worker(problem_spec: Dict, constants: Constants) -> None:
    global _PROBLEM, _CONSTANTS
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _PROBLEM = build_problem(**problem_spec)
    _CONSTANTS = constants
    # One live log per worker process; renamed to the final name after each case.
    _PROBLEM.redirect_log(str(LOG_DIR / f"New_{os.getpid()}.txt"))


def run_case(case: Case, iteration: int) -> Tuple[str, str, float]:
    label, x_0, radius, method, gh_type = case
    miu, theta, shrink, extend, _, p, _, _ = _CONSTANTS

    # A previous case that crashed in this worker leaves a stale live log
    # behind; drop it so its lines don't leak into this case's log.
    live = LOG_DIR / f"New_{os.getpid()}.txt"
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
        max_iter=iteration,
        gh_type=gh_type,
    )
    f_final = float(_PROBLEM.output(result))
    _PROBLEM.flush_log()

    name = log_name(label, x_0, radius, p, method, gh_type)
    live.rename(LOG_DIR / f"{name}.txt")
    return label, name, f_final


def run_suite(
    problem_spec: Dict,
    constants: Constants,
    cases: List[Case],
    iteration: int,
    max_workers: Optional[int] = MAX_WORKERS,
) -> None:
    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=_init_worker, initargs=(problem_spec, constants)
    ) as pool:
        futures = {pool.submit(run_case, case, iteration): case[0] for case in cases}
        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                label, name, f_final = future.result()
            except Exception as exc:  # one bad case should not kill the suite
                print(f"[{done}/{len(cases)}] {futures[future]}: FAILED ({exc})")
                continue
            print(f"[{done}/{len(cases)}] {label}: f = {f_final:.6g}  ({name})")


def run_experiment(experiment: Experiment, max_workers: Optional[int] = MAX_WORKERS) -> None:
    """Report, clear the experiment's old logs, and run every case on the pool."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Details and distribution of what is about to run: printed up front and
    # saved next to the log folder so it can be inspected after the run too.
    summary = describe_cases(experiment.cases, experiment.iteration)
    print(summary)
    (LOG_DIR.parent / f"{experiment.name}_case_distribution.txt").write_text(summary + "\n")

    # Start from a clean slate so the suite ends with exactly len(cases) logs —
    # but only for this experiment's own prefix, since other suites share the
    # folder and their logs must survive.
    for stale in LOG_DIR.glob(f"{experiment.prefix}*.txt"):
        stale.unlink()
    for orphan in LOG_DIR.glob("New*.txt"):
        orphan.unlink()

    run_suite(
        experiment.problem,
        experiment.constants,
        experiment.cases,
        experiment.iteration,
        max_workers,
    )
