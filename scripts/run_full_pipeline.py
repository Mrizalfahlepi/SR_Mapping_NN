"""
SR_Mapping_NN — run_full_pipeline.py
======================================
Master script that runs all 8 pipeline steps in order:

  Step 01 — Download XAUUSD data (H1, D1)
  Step 02 — Feature engineering (26 features)
  Step 03 — Labeling (TP/SL scan, MFE/MAE)
  Step 04 — Train XGBoost model (3 configs)
  Step 05 — Export to ONNX
  Step 06 — Generate charts (4 plots)
  Step 07 — Equity simulation (Original EA vs EA+NN)

Each step imports and calls main() from the corresponding script.
Errors in any step are caught, reported, and do not abort subsequent steps
(unless a later step depends on the failed step's output).

Usage:
    python run_full_pipeline.py [--skip-download]

  --skip-download : Skip step 01 (useful when data already exists)

Output summary is printed at the end with timing for each step.
"""

import argparse
import importlib.util
import os
import sys
import time
import traceback

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Ensure scripts directory is on path so we can import siblings
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


# ---------------------------------------------------------------------------
# Dynamic module loader (avoids polluting sys.modules with numbered names)
# ---------------------------------------------------------------------------
def load_module(filename: str):
    """Load a script file as a Python module by its filename."""
    path = os.path.join(SCRIPT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Script not found: {path}")
    module_name = filename.replace(".py", "").replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------
def run_step(step_num: int, filename: str, description: str,
             results: dict) -> tuple[bool, float]:
    """
    Run a single pipeline step.
    Returns (success: bool, elapsed_seconds: float).
    """
    sep = "=" * 65
    banner = f"  STEP {step_num:02d}: {description}  "
    print(f"\n{sep}")
    print(f"{banner}")
    print(f"{sep}")

    t_start = time.time()
    try:
        module = load_module(filename)
        if not hasattr(module, "main"):
            raise AttributeError(f"{filename} has no main() function")
        module.main()
        elapsed = time.time() - t_start
        results[step_num] = {"status": "OK", "elapsed": elapsed, "description": description}
        print(f"\n  [STEP {step_num:02d} COMPLETE]  {elapsed:.1f}s")
        return True, elapsed

    except Exception as exc:
        elapsed = time.time() - t_start
        results[step_num] = {
            "status": "FAILED",
            "elapsed": elapsed,
            "description": description,
            "error": str(exc),
        }
        print(f"\n  [STEP {step_num:02d} FAILED]  {elapsed:.1f}s")
        print(f"  Error: {exc}")
        print("  Traceback:")
        traceback.print_exc()
        return False, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="SR_Mapping_NN — Full Pipeline Runner"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip step 01 (data download) if data files already exist",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        metavar="N",
        help="Start from step N (1–7). E.g. --start-from 4 skips steps 1-3.",
    )
    parser.add_argument(
        "--only",
        type=int,
        default=None,
        metavar="N",
        help="Run only step N and exit.",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("SR_Mapping_NN — Full Pipeline")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Scripts dir:  {SCRIPT_DIR}")
    print("=" * 65)

    pipeline_start = time.time()

    # Pipeline definition
    steps = [
        (1, "01_download_data.py",      "Download XAUUSD H1 + D1 data"),
        (2, "02_feature_engineering.py","Compute 26 features"),
        (3, "03_labeling.py",           "Label trades (TP/SL scan)"),
        (4, "04_train_model.py",        "Train XGBoost model"),
        (5, "05_export_onnx.py",        "Export to ONNX"),
        (6, "06_generate_charts.py",    "Generate charts"),
        (7, "07_equity_simulation.py",  "Equity simulation"),
    ]

    results = {}

    for (step_num, filename, description) in steps:
        # Skip logic
        if args.only is not None and step_num != args.only:
            results[step_num] = {"status": "SKIPPED", "elapsed": 0.0,
                                  "description": description}
            continue

        if step_num < args.start_from:
            results[step_num] = {"status": "SKIPPED", "elapsed": 0.0,
                                  "description": description}
            print(f"  Skipping step {step_num:02d} (--start-from {args.start_from})")
            continue

        if step_num == 1 and args.skip_download:
            # Check if data already exists
            data_dir = os.path.join(PROJECT_ROOT, "data")
            h1_exists = os.path.exists(os.path.join(data_dir, "xauusd_h1.csv"))
            d1_exists = os.path.exists(os.path.join(data_dir, "xauusd_d1.csv"))
            if h1_exists and d1_exists:
                print(f"\n  [STEP 01 SKIPPED]  --skip-download flag set, data files found")
                results[step_num] = {"status": "SKIPPED", "elapsed": 0.0,
                                      "description": description}
                continue
            else:
                print(f"\n  Note: --skip-download set but data files not found; running download")

        success, elapsed = run_step(step_num, filename, description, results)

        # Hard stop if a critical dependency fails
        if not success and step_num in (1, 2, 3, 4):
            print(f"\n  Critical step {step_num:02d} failed — subsequent steps may fail.")
            print("  Continuing anyway…")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    total_elapsed = time.time() - pipeline_start
    print("\n" + "=" * 65)
    print("Pipeline Summary")
    print("=" * 65)
    print(f"\n  {'Step':<6}  {'Status':<10}  {'Time':>8}  Description")
    print("  " + "-" * 60)

    all_ok = True
    for (step_num, _, _) in steps:
        r = results.get(step_num, {"status": "NOT RUN", "elapsed": 0.0,
                                    "description": ""})
        status = r["status"]
        elapsed = r["elapsed"]
        desc = r["description"]
        icon = {
            "OK": "✓", "FAILED": "✗", "SKIPPED": "–", "NOT RUN": "?"
        }.get(status, "?")
        print(f"  {step_num:>4}   {icon} {status:<9}  {elapsed:>6.1f}s  {desc}")
        if status == "FAILED":
            all_ok = False
            err = r.get("error", "")
            if err:
                print(f"         Error: {err[:80]}")

    print(f"\n  Total time: {total_elapsed:.1f}s")
    overall = "SUCCESS" if all_ok else "COMPLETED WITH ERRORS"
    print(f"  Result:     {overall}")
    print("=" * 65)

    # List generated files
    for sub in ("data", "models", "configs", "charts"):
        d = os.path.join(PROJECT_ROOT, sub)
        if os.path.isdir(d):
            files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
            if files:
                print(f"\n  [{sub}/]")
                for fn in sorted(files):
                    fp = os.path.join(d, fn)
                    print(f"    {fn:<40} {os.path.getsize(fp):>10,} bytes")

    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
