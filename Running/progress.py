"""How far each experiment's current run has progressed.

Every finished case leaves exactly one final-named log in the run directory
(see Optimize.pending_cases), so progress is simply declared cases minus
pending ones — exact, and safe to run any time from Windows or WSL without
disturbing a suite in flight. The ETA comes from the completion times (file
mtimes) of the most recent finished cases, so it stays honest across resumes
and sleep gaps instead of averaging over them.

A suite with cases left whose newest log is old has probably stopped (sleep
cut the terminal, crash); the report says so instead of showing an ETA, which
is the after-wake signal to relaunch the entry script and let it resume.

Usage:  python Running/progress.py            one snapshot
        watch -n 60 python Running/progress.py   refresh every minute (WSL)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

from general_model.Optimize import LOG_ROOT, _dir_name, pending_cases

import Running.smooth_four_methods as smooth_four_methods
import Running.nonsmooth_gh_compare as nonsmooth_gh_compare
import Running.nonsmooth_method_compare as nonsmooth_method_compare

# How many of the newest completions the rate estimate uses. Small enough to
# ignore pauses (sleep, resume) before the recent stretch, large enough to span
# several waves of the ~16 parallel workers, whose completions land in bursts.
RATE_WINDOW = 60
# With ~16 workers a live suite finishes cases far more often than this; a
# quiet stretch this long with cases still pending means the run has stopped.
STALLED_AFTER_S = 10 * 60


def report(experiment) -> str:
    # LOG_ROOT joined directly (not run_dir_for) so a progress check never
    # creates a run directory that no suite has written to yet.
    run_dir = LOG_ROOT / f"{experiment.evaluations} Evalu" / _dir_name(experiment.name)
    total = len(experiment.cases)
    left = len(pending_cases(experiment, run_dir)) if run_dir.is_dir() else total
    done = total - left

    line = (
        f"{experiment.name} @ {experiment.evaluations} evals: "
        f"{done}/{total} cases done ({100 * done // total}%)"
    )
    if left == 0:
        return line + "  COMPLETE"
    if done == 0:
        return line + "  (not started)"

    mtimes = sorted(
        f.stat().st_mtime
        for f in run_dir.glob(f"{experiment.prefix}*.txt")
    )[-RATE_WINDOW:]
    quiet = time.time() - mtimes[-1]
    if quiet > STALLED_AFTER_S:
        return line + (
            f"  no new logs for {quiet / 60:.0f} min - probably not running; "
            f"relaunch the entry script to resume"
        )
    if len(mtimes) >= 2 and mtimes[-1] > mtimes[0]:
        rate = (len(mtimes) - 1) / (mtimes[-1] - mtimes[0])  # cases per second
        eta_min = left / rate / 60
        eta = "<1 min" if eta_min < 1 else f"~{eta_min:.0f} min"
        line += f"  {eta} left at the recent rate"
    return line


if __name__ == "__main__":
    for module in (smooth_four_methods, nonsmooth_gh_compare, nonsmooth_method_compare):
        print(report(module.EXPERIMENT))
