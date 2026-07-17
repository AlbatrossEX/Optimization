"""Shared helpers for the BQmin Monte-Carlo studies (BQ_subcompare.py and
generate_top10percent_analysis.py).

Both studies draw random points and radii and, per trial, compare the smooth
objective at three candidates: the origin, the best interpolation (poised) point,
and the bqmin step. The per-trial comparison and the problem setup are identical,
so they live here; each study keeps only its own trial count, binning and plots.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # project root
sys.path.insert(0, str(ROOT))

import numpy as np

from construct_functions import build_smooth_problem
from general_model.Smooth.models.bqmin import bqmin

GRAPH_DIR = Path(__file__).resolve().parent / "Graphs"  # generated figures land here


def build_study_problem(m: int = 15, nprob: int = 8, quiet: bool = True):
    """The study problem (Bard, nprob 8 -- same as the smooth suite).

    quiet=True routes the model-build evaluation log to the null device, so the
    study does not drop a stray New.txt into Log/Logs (it never reads the log
    back; it only needs the returned f values)."""
    problem = build_smooth_problem(m=m, nprob=nprob)
    if quiet:
        problem.redirect_log(os.devnull)
    return problem


def compare_once(problem, x, radius, dim):
    """(f at origin, best f over interpolation points, f at bqmin step) for one trial.

    The interpolation minimum is free once GH has built the model; the bqmin step
    costs one extra evaluation on top of that model."""
    f_origin = float(problem.f(x))
    g, h = problem.GH(x, radius)
    f_interp = float(np.min(problem.f_poised))
    bound = radius * np.ones(dim)
    step, _ = bqmin(h, g, -bound, bound)
    f_bq = float(problem.f(x + np.asarray(step, dtype=float)))
    return f_origin, f_interp, f_bq
