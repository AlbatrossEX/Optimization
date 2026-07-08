import atexit
import queue
import threading
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Optional, Tuple
from .bqmin import bqmin

Array1D = NDArray[np.floating]


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
    def __init__(self, f: Callable[[Array1D], np.floating]):
        self.f = f
        self.count = 0
        self._logger = _AsyncLogWriter("Log/Logs/New.txt")

    def output(self, input: Array1D) -> np.floating:
        self.count += 1
        value = self.f(input)
        self._logger.write(f"{self.count},{input},{value},\n")
        return value

    def flush_log(self) -> None:
        """Wait for all pending async log writes to hit disk (e.g. before renaming the log file)."""
        self._logger.flush()

    def GH(self, x: Array1D, radius: float) -> Tuple[Array1D, Array1D]:
        raise NotImplementedError

    def trust_region_optimization(
        self,
        x_0: Array1D,
        miu: float,
        theta: float,
        shrink: float,
        extend: float,
        radius: float,
        p: float,
        max_iter: int = 1000,
    ) -> Array1D:
        x = np.array(x_0, dtype=float, copy=True)
        delta = float(radius)
        fx = float(self.f(x))

        for _ in range(max_iter):
            g, h = self.GH(x, delta)
            bound = delta * np.ones_like(x)
            step, _ = bqmin(h, g, -bound, bound)
            step = np.asarray(step, dtype=float).reshape(x.shape)

            predicted_reduction = float(-(g @ step + 0.5 * step @ h @ step))
            if predicted_reduction <= 0:
                delta *= shrink
                continue

            f_trial = float(self.f(x + step))
            actual_reduction = fx - f_trial
            if not np.isfinite(actual_reduction):
                delta *= shrink
                continue
            roll = actual_reduction / (theta * (np.linalg.norm(step, 2) ** (1.0 + p)))

            if roll >= miu:
                x = x + step
                fx = f_trial
                if roll > 0.75:
                    delta *= extend
            else:
                delta *= shrink

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
        max_iter: int = 1000,
    ) -> Array1D:
        x = np.array(x_0, dtype=float, copy=True)
        delta = float(radius)

        for _ in range(max_iter):
            g, h = self.GH(x,delta)
            lower = -delta * np.ones_like(x)
            upper = delta * np.ones_like(x)
            step, _ = bqmin(h, g, lower, upper)
            step = np.asarray(step, dtype=float).reshape(x.shape)

            afterstep:float = self.f(x + step)
            min_interp:float = np.min(self.f_poised)

            if afterstep >= min_interp:
                afterstep = min_interp
                flat_idx = np.argmin(self.f_poised)
                step = self.poised[flat_idx, :] - x
                continue

            actual_reduction = float(self.f(x) - afterstep)
            if not np.isfinite(actual_reduction):
                delta *= shrink
                continue
            roll = actual_reduction / (theta * (np.linalg.norm(step, 2) ** (1.0 + p)))

            if roll >= miu:
                x = x + step
                if roll > 0.75:
                    delta *= extend
            else:
                delta *= shrink

        return x


