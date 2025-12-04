#!/usr/bin/env python3
"""
Grid-search-capable wrapper to run TCDF (s_TCDF) across kernel==dilation values
and varying hidden layers. Designed to run as a single job/pod that executes
many TCDF runs serially and saves results to an output directory (e.g. on your PVC).

How to run (example):
    python grid_search_find_correct_lag.py \
      --datafile /data/cleaned_linear_ts_n500_vars4_lag3_gaussian.csv \
      --kernels 2 3 4 \
      --hidden_layers 0 1 2 \
      --desired_lag 3 \
      --epochs 200 \
      --cuda False \
      --outdir /data/results/grid_search_run

This script:
 - Loops over kernel_size values (and forces dilation_coefficient == kernel_size).
 - Loops over hidden_layers (int). The script passes layers = hidden_layers + 1 to s_TCDF.findcauses
   (this matches the original TCDF design where levels = hidden_layers + 1).
 - Skips combos whose receptive field is smaller than desired_lag (optional).
 - Runs s_TCDF.findcauses for each target column and saves per-combo JSON results.
 - Prints progress and writes failures into the output JSON for later inspection.
"""

import sys
from pathlib import Path
import importlib.util
import argparse
import json
import time
from typing import Dict, Tuple, Any, List

# # # Adjust these repo_root paths if you run the script from a different location.
# repo_root = Path("/home/kenny/Documents/repo_github/Modified_TCDF").resolve()
# repo_root2 = Path("/home/kenny/Documents/repo_github/synth_timeseries_data").resolve()

# # Ensure repo roots are on sys.path so local imports (s_TCDF, helper_funcs) work.
# sys.path.insert(0, str(repo_root))
# sys.path.insert(0, str(repo_root2))

import os
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# # local project helpers (must be importable from repo_root)
# import s_TCDF
# from helper_funcs import metrics_helper, tcdf_helper

try:
    import s_TCDF
    from helper_funcs import metrics_helper, tcdf_helper
except ImportError as e:
    raise ImportError(
        "Failed to import project modules (s_TCDF / helper_funcs). "
        "Set PYTHONPATH to include your repo directories or install the packages. "
        "Example: export PYTHONPATH=/data/repo_github/Modified_TCDF:/data/repo_github/synth_timeseries_data:$PYTHONPATH"
    ) from e

# Reproducibility seeds
GLOBAL_SEED = 1111
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(GLOBAL_SEED)


def receptive_field(kernel_size: int, dilation_c: int, hidden_layers: int) -> int:
    """Compute receptive field (same formula as runTCDF.py).
    levels = hidden_layers + 1
    rf = 1 + sum_{l=0..levels-1} (kernel_size - 1) * (dilation_c**l)
    """
    levels = hidden_layers + 1
    rf = 1
    for l in range(levels):
        rf += (kernel_size - 1) * (dilation_c ** l)
    return rf


def run_tcdf_file(datafile: str, params: Dict) -> Tuple[Dict, Dict, Dict, Dict, List[str]]:
    """
    Run TCDF (s_TCDF.findcauses) for every column in datafile according to params.
    Returns (allcauses, alldelays, allreallosses, allscores, columns)
    """
    # read csv once
    df_data = pd.read_csv(datafile)
    columns = list(df_data.columns)

    allcauses = {}
    alldelays = {}
    allreallosses = {}
    allscores = {}

    # translate params
    kernel_size = params.get("kernel_size", 4)
    hidden_layers = params.get("hidden_layers", 0)
    # TCDF.findcauses historically expects 'layers' == hidden_layers + 1
    layers_arg = hidden_layers + 1
    epochs = params.get("epochs", 1000)
    lr = params.get("learning_rate", 0.01)
    optimizername = params.get("optimizer", "Adam")
    log_interval = params.get("log_interval", 500)
    seed = params.get("seed", GLOBAL_SEED)
    dilation_c = params.get("dilation_coefficient", kernel_size)
    significance = params.get("significance", 0.8)
    cuda_flag = params.get("cuda", False)

    def _call_findcauses(target_col):
        """Try to call s_TCDF.findcauses with df= to avoid re-reading; fallback to file path."""
        kwargs = dict(
            c=target_col,
            cuda=cuda_flag,
            epochs=epochs,
            kernel_size=kernel_size,
            layers=layers_arg,
            log_interval=log_interval,
            lr=lr,
            optimizername=optimizername,
            seed=seed,
            dilation_c=dilation_c,
            significance=significance,
        )
        try:
            # prefer df-based API if available
            return s_TCDF.findcauses(df=df_data, **kwargs)
        except TypeError:
            # fallback to older API that reads file from disk
            kwargs["file"] = str(datafile)
            # s_TCDF.findcauses likely expects target as the first positional arg
            return s_TCDF.findcauses(target_col, **{k: v for k, v in kwargs.items() if k != "c"})

    iterator = tqdm(columns, desc="targets") if tqdm is not None else columns
    for idx, c in enumerate(iterator):
        try:
            causes, causeswithdelay, realloss, scores = _call_findcauses(c)
        except Exception as e:
            # On failure for this target, record the error and continue
            print(f"Error running findcauses for target {c}: {e}")
            allscores[idx] = []
            allcauses[idx] = []
            allreallosses[idx] = None
            continue

        allscores[idx] = scores
        allcauses[idx] = causes
        alldelays.update(causeswithdelay)
        allreallosses[idx] = realloss

    return allcauses, alldelays, allreallosses, allscores, columns


def main():
    p = argparse.ArgumentParser(description="Grid search wrapper for TCDF (run multiple combos in one pod).")
    p.add_argument("--datafile", required=True, help="Path to CSV dataset (columns = variables).")
    p.add_argument("--kernels", type=int, nargs="+", required=True, help="kernel_size values to try (dilation will be set equal).")
    p.add_argument("--hidden_layers", type=int, nargs="+", required=True, help="hidden_layers values to try (levels = hidden_layers+1).")
    p.add_argument("--desired_lag", type=int, default=None, help="Optional: skip combos whose receptive field < desired_lag.")
    p.add_argument("--epochs", type=int, default=200, help="Epochs to run each TCDF instance (use small for quick grid).")
    p.add_argument("--cuda", action="store_true", help="Use CUDA if available.")
    p.add_argument("--outdir", default="./results", help="Directory to write per-combo result JSON files.")
    p.add_argument("--log_interval", type=int, default=1000)
    p.add_argument("--learning_rate", type=float, default=0.01)
    p.add_argument("--optimizer", type=str, default="Adam")
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    p.add_argument("--significance", type=float, default=0.8)
    args = p.parse_args()

    # ensure outdir exists
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # sanity for CUDA flag
    if args.cuda and not torch.cuda.is_available():
        print("WARNING: --cuda requested but no CUDA available; running on CPU instead.")
        use_cuda = False
    else:
        use_cuda = args.cuda and torch.cuda.is_available()

    combos_summary = {}

    for k in args.kernels:
        for hl in args.hidden_layers:
            dilation = k  # enforce kernel == dilation
            rf = receptive_field(kernel_size=k, dilation_c=dilation, hidden_layers=hl)
            combo_name = f"k{k}_hl{hl}_rf{rf}"
            combo_out = {
                "kernel_size": k,
                "dilation": dilation,
                "hidden_layers": hl,
                "receptive_field": rf,
                "skipped": False,
                "start_time": None,
                "end_time": None,
                "results_file": None,
                "error": None
            }

            # optionally skip combos that cannot cover desired lag
            if args.desired_lag is not None and rf < args.desired_lag:
                print(f"Skipping combo {combo_name}: receptive field {rf} < desired_lag {args.desired_lag}")
                combo_out["skipped"] = True
                combos_summary[combo_name] = combo_out
                continue

            print(f"\n=== Running combo: {combo_name} ===")
            params = {
                "cuda": use_cuda,
                "epochs": args.epochs,
                "kernel_size": k,
                "hidden_layers": hl,
                "learning_rate": args.learning_rate,
                "optimizer": args.optimizer,
                "log_interval": args.log_interval,
                "seed": args.seed,
                "dilation_coefficient": dilation,
                "significance": args.significance,
            }

            combo_out["start_time"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

            try:
                allcauses, alldelays, allreallosses, allscores, columns = run_tcdf_file(args.datafile, params)
                graph_matrix, val_matrix = tcdf_helper.make_matrices(alldelays, columns, allscores)
                print(f"Discovered graph matrix shape: {graph_matrix.shape}")

                combo_out['causal_graph_shape'] = f'vars:{graph_matrix.shape[0]},lags:{graph_matrix.shape[2]}'
                # Build a lightweight result dict to persist
                discovered = []
                for key, delay in alldelays.items():
                    # key is (effect_idx, cause_idx)
                    discovered.append({"effect_idx": int(key[0]), "cause_idx": int(key[1]), "delay": int(delay)})

                # Convert numpy types to native Python for JSON serialization
                serializable = {
                    "params": params,
                    "columns": columns,
                    "validated_causes_counts": {str(k): len(v) for k, v in allcauses.items()},
                    "discovered_delays": discovered,
                    "losses": {str(k): (None if v is None else float(v)) for k, v in allreallosses.items()},
                    "scores_shapes": {str(k): (0 if v is None else len(v)) for k, v in allscores.items()},
                }

                timestamp = int(time.time())
                outpath = outdir / f"tcdf_grid_{combo_name}_{timestamp}.json"
                with open(outpath, "w") as fh:
                    json.dump(serializable, fh, indent=2)

                combo_out["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
                combo_out["results_file"] = str(outpath)

                # quick check if desired lag was found anywhere
                if args.desired_lag is not None:
                    combo_out["found_desired_lag"] = any(int(d["delay"]) == args.desired_lag for d in discovered)
                else:
                    combo_out["found_desired_lag"] = None

                print(f"Combo finished — results saved to {outpath}")

            except Exception as e:
                combo_out["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
                combo_out["error"] = str(e)
                print(f"Error while running combo {combo_name}: {e}")

            combos_summary[combo_name] = combo_out

            # small pause to avoid hammering I/O at job start
            time.sleep(1.0)

    # Write summary of combos
    summary_path = outdir / f"tcdf_grid_summary_{int(time.time())}.json"
    with open(summary_path, "w") as fh:
        json.dump(combos_summary, fh, indent=2)
    print(f"\nGrid search finished. Summary written to {summary_path}")


if __name__ == "__main__":
    main()

