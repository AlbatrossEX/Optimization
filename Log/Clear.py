"""Manual utility to wipe generated run logs.

Runs are NEVER cleared automatically any more: each run of a Running/ experiment
writes into its own Log/Logs/<name>_<timestamp>/ directory and they accumulate.
Run this script by hand only when you deliberately want to reclaim that space.

  py Log/Clear.py            # delete every run directory and any loose .txt logs
  py Log/Clear.py smooth     # only run directories whose name starts with "smooth"

IMPORTANT: all deletion is guarded under __main__, so importing this module NEVER
deletes anything. Only running it as a script does.
"""
import os
import shutil
import sys


def clear(prefix: str = "") -> None:
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Logs")
    if not os.path.isdir(folder):
        print(f"no log folder at {folder}")
        return

    removed = 0
    for entry in os.listdir(folder):
        path = os.path.join(folder, entry)
        if prefix and not entry.startswith(prefix):
            continue
        if os.path.isdir(path):  # a per-run directory
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
        elif entry.endswith(".txt"):  # loose logs from the old flat layout
            os.remove(path)
            removed += 1

    print(f"removed {removed} item(s) from {folder}"
          + (f" matching '{prefix}*'" if prefix else ""))


if __name__ == "__main__":
    clear(sys.argv[1] if len(sys.argv) > 1 else "")
