import atexit
import queue
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from numpy.typing import NDArray
from pathlib import Path
from typing import Callable, Optional, Tuple
from .bqmin import bqmin

Array1D = NDArray[np.floating]

# anchored to the project root so logging works regardless of the caller's cwd
_LIVE_LOG = str(Path(__file__).resolve().parents[1] / "Log" / "Logs" / "New.txt")

# Trust-radius floor: once the radius collapses below this the model box is
# numerically a point and no step can make progress, so the solver has stalled.
# Stalling is NOT an early stop for the run: the stopping condition is the
# function-evaluation budget (max_evals), and every log must carry at least
# that many evaluations so runs stay comparable point-for-point. A stalled
# solver therefore leaves its iteration loop (which could otherwise shrink
# forever without spending an evaluation) and spends whatever remains of the
# budget at its final iterate via _spend_remaining.
_DELTA_MIN = 1e-12


class _AsyncLogWriter:
    """Writes log lines from a background thread so callers never block on disk I/O."""

    def __init__(self, path: str):
        self._path = path
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        atexit.register(self.stop)

    def _run(self) -> None:
        # Opened per batch (not held open) so the log file can be renamed/deleted
        # by the main thread between writes without fighting the writer for the handle.
        # Draining the whole backlog per open matters: one open/close per line is
        # what dominates runtime on slow filesystems (SynologyDrive, /mnt/c).
        while True:
            batch = [self._queue.get()]
            while True:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            lines = [l for l in batch if l is not None]
            if lines:
                with open(self._path, "a") as f:
                    f.writelines(lines)
            for _ in batch:
                self._queue.task_done()
            if None in batch:
                break

    def write(self, line: str) -> None:
        self._queue.put(line)

    def flush(self) -> None:
        """Block until every queued line so far has been written to disk."""
        self._queue.join()

    def stop(self) -> None:
        self._queue.put(None)
        self._thread.join()


class TR_function:
    # How many threads price a whole interpolation (poised) set at once (see
    # _evaluate_poised). 1 = serial. The parallel suite sets this to 1 per worker
    # because it already fans out one process per pipeline, so per-worker threads
    # would only oversubscribe the cores. A single-process caller (e.g. an ad-hoc
    # run or Log/BQmin_graphing/BQ_subcompare.py) can raise it to overlap the
    # poised evaluations across idle cores.
    _eval_workers: int = 1

    def __init__(self, f: Callable[[Array1D], np.floating]):
        self.f = f
        self.count = 0
        self._logger = _AsyncLogWriter(_LIVE_LOG)
        self._pool: Optional[ThreadPoolExecutor] = None  # lazily built poised pool

    def output(self, input: Array1D) -> np.floating:
        self.count += 1
        value = self.f(input)
        self._logger.write(f"{self.count},{input},{value},\n")
        return value

    def _evaluate_poised(self, points: Array1D) -> Array1D:
        """f-value of each row of `points`, returned as a 1-D array.

        When _eval_workers > 1 the objective values are computed concurrently on
        a reused thread pool — calfun is numpy-heavy, so its array work overlaps
        across threads — and only then folded into the shared evaluation counter
        and the log, in row order. Separating the compute from the counting keeps
        self.count and the log byte-for-byte identical to the serial path however
        many threads ran, so a run's logs never depend on _eval_workers.
        """
        points = np.asarray(points, dtype=float)
        n = points.shape[0]
        if n == 0:
            return np.empty(0)

        workers = min(int(self._eval_workers), n)
        if workers > 1:
            # One pool per problem object, reused across every poised set, so the
            # threads are not re-spawned on each GH/iteration call.
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=int(self._eval_workers))
            values = list(self._pool.map(lambda row: self.f(row), points))
        else:
            values = [self.f(points[i, :]) for i in range(n)]

        out = np.empty(n)
        for i, value in enumerate(values):
            self.count += 1
            self._logger.write(f"{self.count},{points[i, :]},{value},\n")
            out[i] = float(value)
        return out

    def flush_log(self) -> None:
        """Wait for all pending async log writes to hit disk (e.g. before renaming the log file)."""
        self._logger.flush()

    def redirect_log(self, path: str) -> None:
        """Route subsequent log lines to a new file (each parallel worker needs its own)."""
        old = self._logger
        self._logger = _AsyncLogWriter(path)
        old.stop()

    def GH(self, x: Array1D, radius: float, gh_type: int = 0) -> Tuple[Array1D, Array1D]:
        raise NotImplementedError

    def _resize_radius(self, delta: float, factor: float, action: str) -> float:
        """Scale the trust radius and record the change in the log.

        The line reads radius,<evaluation count>,<shrink|extend>,<old>,<new>, —
        it starts with 'radius' so the graph parsers' evaluation-line pattern
        (which requires a leading evaluation number) skips it, while the count
        ties the change to the log's evaluation axis for later analysis.
        """
        resized = delta * factor
        self._logger.write(f"radius,{self.count},{action},{delta:g},{resized:g},\n")
        return resized

    def _spend_remaining(self, x: Array1D, start_count: int, budget: int) -> None:
        """Spend whatever is left of the evaluation budget at the final iterate.

        A stalled solver (collapsed trust radius) leaves its iteration loop
        before the budget is used up; the budget is the stopping condition and
        each log must contain at least `budget` evaluations, so the remainder
        is spent (and logged) here at the point the solver stalled on.
        """
        while self.count - start_count < budget:
            self.output(x)

    # --- Solvers -------------------------------------------------------------
    # Every solver stops on a FUNCTION-EVALUATION budget, not an iteration count:
    # `max_evals` is how many times the objective may be evaluated. self.count is
    # the shared evaluation counter, so a run is bounded by how far
    # `self.count - start_count` is allowed to grow. The budget is checked at the
    # top of each iteration, so the final iteration may carry the count slightly
    # past the budget (by one iteration's worth of evaluations); this is the
    # standard "stop once the budget is spent" semantics.

    def trust_region_optimization_0(
        self,
        x_0: Array1D,
        miu: float,
        theta: float,
        shrink: float,
        extend: float,
        radius: float,
        p: float,
        max_evals: int,
        gh_type: int,
    ) -> Array1D:
        x = np.array(x_0, dtype=float, copy=True)
        delta = float(radius)
        budget = int(max_evals)
        start_count = self.count
        # output() (not self.f) so the solver's own evaluations are counted and logged
        fx = float(self.output(x))

        while self.count - start_count < budget:
            g, h = self.GH(x, delta, gh_type)
            bound = delta * np.ones_like(x)
            step, _ = bqmin(h, g, -bound, bound)
            step = np.asarray(step, dtype=float).reshape(x.shape)

            f_trial = float(self.output(x + step))
            actual_reduction = fx - f_trial

            if not np.isfinite(fx) and np.isfinite(f_trial):
                x = x + step
                fx = f_trial
                continue

            if not (np.isfinite(fx) or np.isfinite(f_trial)):
                delta = self._resize_radius(delta, extend, "extend")
                continue

            roll = actual_reduction / (theta * (np.linalg.norm(step, 2) ** (1.0 + p)))

            if roll >= miu:
                x = x + step
                fx = f_trial
                delta = self._resize_radius(delta, extend, "extend")
            else:
                delta = self._resize_radius(delta, shrink, "shrink")

        return x


    def trust_region_optimization_1(
        self,
        x_0: Array1D,
        miu: float,
        theta: float,
        shrink: float,
        extend: float,
        radius: float,
        p: float,
        max_evals: int = 1000,
        gh_type: int = 0,
    ) -> Array1D:
        """Steps to the best interpolation (poised) point.

        This solver does NOT call GH: the best-interpolation-point strategy needs
        only the poised geometry, not a fitted quadratic model, so it builds the
        interpolation set straight from algorithm_6_4 and skips the fitfroquad
        model fit entirely. gh_type is therefore unused here (the poised set is
        the interpolation geometry, i.e. the gh_type-0 model).

        Like the dynamic solver, fx is already known, so the set is built without
        spending an evaluation on the centre point.
        """
        # Imported here, not at module level: the classes that own the smooth
        # machinery import this module, so a top-level import could go circular.
        from .Smooth.algorism6_4 import algorithm_6_4

        x = np.array(x_0, dtype=float, copy=True)
        delta = float(radius)
        budget = int(max_evals)
        start_count = self.count
        # output() (not self.f) so the solver's own evaluations are counted and logged
        fx = float(self.output(x))

        while self.count - start_count < budget:
            if delta < _DELTA_MIN:
                break  # trust region collapsed; the remaining budget is spent below
            offsets, _ = algorithm_6_4(
                Y=np.zeros_like(x.reshape(1, -1)),
                Delta=delta,
                f=np.array([[fx]]),
            )
            poised = offsets + x
            # The centre (zero offset) sits in the poised set; its value is fx and
            # stepping there is a no-op, so only the other points are candidates.
            candidates = np.array(
                [pt for pt in poised if np.linalg.norm(pt - x, 2) > 0.0]
            )
            if candidates.shape[0] == 0:
                delta = self._resize_radius(delta, shrink, "shrink")
                continue

            f_candidates = self._evaluate_poised(candidates)
            best_idx = int(np.argmin(f_candidates))
            f_trial = float(f_candidates[best_idx])
            step = candidates[best_idx, :] - x

            actual_reduction = fx - f_trial

            if not np.isfinite(fx) and np.isfinite(f_trial):
                x = x + step
                fx = f_trial
                continue

            if not (np.isfinite(fx) or np.isfinite(f_trial)):
                delta = self._resize_radius(delta, extend, "extend")
                continue

            roll = actual_reduction / (theta * (np.linalg.norm(step, 2) ** (1.0 + p)))

            if roll >= miu:
                x = x + step
                fx = f_trial
                delta = self._resize_radius(delta, extend, "extend")
            else:
                delta = self._resize_radius(delta, shrink, "shrink")

        self._spend_remaining(x, start_count, budget)
        return x

    def trust_region_optimization_2(
        self,
        x_0: Array1D,
        miu: float,
        theta: float,
        shrink: float,
        extend: float,
        radius: float,
        p: float,
        max_evals: int = 1000,
        gh_type: int = 0,
    ) -> Array1D:
        """Takes whichever candidate is better: the bqmin step or the best interpolation point."""
        x = np.array(x_0, dtype=float, copy=True)
        delta = float(radius)
        budget = int(max_evals)
        start_count = self.count
        # output() (not self.f) so the solver's own evaluations are counted and logged
        fx = float(self.output(x))

        while self.count - start_count < budget:
            if delta < _DELTA_MIN:
                break  # trust region collapsed; the remaining budget is spent below
            g, h = self.GH(x, delta, gh_type)
            bound = delta * np.ones_like(x)
            step, _ = bqmin(h, g, -bound, bound)
            step = np.asarray(step, dtype=float).reshape(x.shape)

            f_trial = float(self.output(x + step))

            # Fall back to the best interpolation point when it beats the model
            # step; its f value was already computed inside GH, so it is free.
            best_idx = int(np.argmin(self.f_poised))
            f_interp = self.f_poised[best_idx].item()
            if f_interp < f_trial:
                f_trial = f_interp
                step = self.poised[best_idx, :] - x

            actual_reduction = fx - f_trial

            if not np.isfinite(fx) and np.isfinite(f_trial):
                x = x + step
                fx = f_trial
                continue

            if not (np.isfinite(fx) or np.isfinite(f_trial)):
                delta = self._resize_radius(delta, extend, "extend")
                continue

            step_norm = float(np.linalg.norm(step, 2))
            # step_norm can be 0 when the best interpolation point is x itself
            if step_norm == 0.0:
                delta = self._resize_radius(delta, shrink, "shrink")
                continue
            roll = actual_reduction / (theta * (step_norm ** (1.0 + p)))

            if roll >= miu:
                x = x + step
                fx = f_trial
                delta = self._resize_radius(delta, extend, "extend")
            else:
                delta = self._resize_radius(delta, shrink, "shrink")

        self._spend_remaining(x, start_count, budget)
        return x

    def trust_region_optimization_3(
        self,
        x_0: Array1D,
        miu: float,
        theta: float,
        shrink: float,
        extend: float,
        radius: float,
        p: float,
        max_evals: int = 1000,
        gh_type: int = 0,
    ) -> Array1D:
        """Dynamic interpolation point: walks the poised set one evaluation at a
        time and moves to the FIRST point that passes the acceptance test,
        abandoning the rest of the set, then rebuilds the set at the new point.

        Method 1 pays for the whole interpolation set before stepping to its
        minimum; this solver stops paying as soon as any poised point is
        acceptable, so a successful step usually costs only a few evaluations.
        A full sweep with no acceptable point shrinks the radius, like the
        other solvers' failed iterations.
        """
        if gh_type != 0:
            raise ValueError(
                "dynamic interpolation point requires the interpolation model (gh_type 0)"
            )
        # Imported here, not at module level: the classes that own the smooth
        # machinery import this module, so a top-level import could go circular.
        from .Smooth.algorism6_4 import algorithm_6_4

        x = np.array(x_0, dtype=float, copy=True)
        delta = float(radius)
        budget = int(max_evals)
        start_count = self.count
        # output() (not self.f) so the solver's own evaluations are counted and logged
        fx = float(self.output(x))

        while self.count - start_count < budget:
            if delta < _DELTA_MIN:
                break  # trust region collapsed; the remaining budget is spent below
            # Geometry only: fx is already known, so unlike GH the set is built
            # without spending an evaluation on the centre point.
            offsets, _ = algorithm_6_4(
                Y=np.zeros_like(x.reshape(1, -1)),
                Delta=delta,
                f=np.array([[fx]]),
            )
            poised = offsets + x

            moved = False
            for point in poised:
                step = point - x
                step_norm = float(np.linalg.norm(step, 2))
                # the centre itself sits in the poised set; its value is fx
                if step_norm == 0.0:
                    continue
                f_trial = float(self.output(point))
                actual_reduction = fx - f_trial

                if not np.isfinite(fx) and np.isfinite(f_trial):
                    # switch immediately: the remaining points are never evaluated
                    x = np.array(point, dtype=float, copy=True)
                    fx = f_trial
                    moved = True
                    break

                if not (np.isfinite(fx) or np.isfinite(f_trial)):
                    continue  # nothing rankable here; the sweep outcome decides

                roll = actual_reduction / (theta * (step_norm ** (1.0 + p)))

                if roll >= miu:
                    # switch immediately: the remaining points are never evaluated
                    x = np.array(point, dtype=float, copy=True)
                    fx = f_trial
                    delta = self._resize_radius(delta, extend, "extend")
                    moved = True
                    break

            if not moved:
                if not np.isfinite(fx):
                    # every point in reach was non-finite too: method 0's rule
                    # for that case is to extend, growing the region to escape
                    delta = self._resize_radius(delta, extend, "extend")
                else:
                    delta = self._resize_radius(delta, shrink, "shrink")

        self._spend_remaining(x, start_count, budget)
        return x

    def trust_region_optimization_function(
        self,
        method: int,
        x_0: Array1D,
        miu: float,
        theta: float,
        shrink: float,
        extend: float,
        radius: float,
        p: float,
        max_evals: int = 1000,
        gh_type: int = 0,
    ) -> Array1D:
        """Dispatches to a solver by method: 0 = bqmin step, 1 = best interpolation
        point, 2 = better of the two, 3 = dynamic interpolation point.

        max_evals is the function-evaluation budget (the stopping condition).

        gh_type selects the model builder inside GH (0 = quadratic interpolation
        fit; NonSmoothFunction additionally supports 1 = random +-1 model, which
        only works with method 0 since methods 1/2/3 need an interpolation set).
        """
        solvers = (
            self.trust_region_optimization_0,
            self.trust_region_optimization_1,
            self.trust_region_optimization_2,
            self.trust_region_optimization_3,
        )
        return solvers[int(method)](
            x_0=x_0,
            miu=miu,
            theta=theta,
            shrink=shrink,
            extend=extend,
            radius=radius,
            p=p,
            max_evals=max_evals,
            gh_type=gh_type,
        )
