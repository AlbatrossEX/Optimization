#!/usr/bin/env python3
"""
Master script to run all optimization experiments and generate graphs sequentially.
"""

import subprocess
import sys
from pathlib import Path

# Define the sequence of runs
RUNS = [
    {
        "name": "Smooth Four Methods",
        "script": "Running/smooth_four_methods.py",
        "graph": "Log/Smooth_four_methods/graph.py",
    },
    {
        "name": "Nonsmooth Method Compare",
        "script": "Running/nonsmooth_method_compare.py",
        "graph": "Log/Nonsmooth_method_compare/graph.py",
    },
    {
        "name": "Nonsmooth GH Compare",
        "script": "Running/nonsmooth_gh_compare.py",
        "graph": "Log/Nonsmooth_gh_compare/graph.py",
    },
]


def run_script(script_path):
    """Run a Python script and return success status."""
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        return False
    except FileNotFoundError:
        print(f"File not found: {script_path}")
        return False


def main():
    """Run all experiments and graphs sequentially."""
    project_root = Path(__file__).parent
    total_runs = len(RUNS)
    successful = 0
    failed = []

    print("=" * 60)
    print("Running All Optimization Experiments and Graphs")
    print("=" * 60)

    for i, run in enumerate(RUNS, 1):
        print(f"\n[{i}/{total_runs}] {run['name']}")
        print("-" * 60)

        # Run the optimization script
        script_path = project_root / run["script"]
        print(f"Running optimization: {run['script']}")
        if not run_script(str(script_path)):
            failed.append(f"{run['name']} (optimization)")
            print(f"⚠ Failed to run {run['script']}")
            continue

        # Run the graph generation script
        graph_path = project_root / run["graph"]
        print(f"Generating graphs: {run['graph']}")
        if not run_script(str(graph_path)):
            failed.append(f"{run['name']} (graphs)")
            print(f"⚠ Failed to run {run['graph']}")
            continue

        successful += 1
        print(f"✓ {run['name']} completed successfully")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Completed: {successful}/{total_runs} experiments")

    if failed:
        print(f"\nFailed runs ({len(failed)}):")
        for fail in failed:
            print(f"  - {fail}")
        sys.exit(1)
    else:
        print("\n✓ All experiments and graphs completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
