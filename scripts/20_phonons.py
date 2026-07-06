#!/usr/bin/env python
"""Stage D1 - phonon dispersion fidelity of the cheap cells vs the fp64 reference.

For each cell (backend/dtype) and displacement amplitude, build force constants
on the canonical monolayer, compute the S-X-Gamma-Y-S dispersion and the minimum
frequency near Gamma (the ZA-branch stability watchdog). Sweeping the
displacement amplitude separates two error sources that a single run conflates:
finite-displacement truncation error (varies with amplitude) vs precision/kernel
error of the cell (visible as the gap to e3nn/float64 at the SAME amplitude).

Outputs: per-(cell, displacement) band npz + json sidecar in results/raw/phonons/
and results/raw/phonons_summary.json with per-branch RMSE vs the reference cell.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from phosbench.common import env_metadata, load_canonical, make_calc, write_json
from phosbench.phonons import (THZ_TO_CM1, band_path_SXGYS, build_phonon,
                               min_freq_near_gamma)

REPO = Path(__file__).resolve().parent.parent
RAW_DIR = REPO / "results" / "raw" / "phonons"
SUMMARY_PATH = REPO / "results" / "raw" / "phonons_summary.json"

REFERENCE_CELL = "e3nn/float64"
IMAG_TOL_THZ = 1e-3            # band point counts as imaginary below -1e-3 THz
DEFAULT_SUPERCELL = "4,6,1"

# mace_mp tags occasionally lag the foundation releases; pin the known URLs.
MODEL_URLS = {
    "medium-mpa-0": ("https://github.com/ACEsuit/mace-foundations/releases/"
                     "download/mace_mpa_0/mace-mpa-0-medium.model"),
    "medium-omat-0": ("https://github.com/ACEsuit/mace-foundations/releases/"
                      "download/mace_omat_0/mace-omat-0-medium.model"),
}


def default_model() -> str:
    """Stage-A winner if present; otherwise the documented zero-shot baseline."""
    try:
        payload = json.loads((REPO / "configs" / "model_choice.json").read_text())
        model = payload if isinstance(payload, str) else payload.get("model")
        if isinstance(model, str) and model:
            return model
        if isinstance(payload, dict) and "medium-omat-0" in payload.get("details", {}):
            print("[phosbench] no Stage-A winner; using medium-omat-0 as the "
                  "documented zero-shot baseline. Pass --model for a fine-tuned "
                  "model.", file=sys.stderr)
            return "medium-omat-0"
    except (OSError, ValueError):
        pass
    return "medium-omat-0"


def make_calc_robust(backend: str, dtype: str, model: str, device: str):
    """make_calc, retrying with the pinned release URL when a tag is unknown."""
    try:
        return make_calc(backend, dtype, model=model, device=device)
    except Exception:
        url = MODEL_URLS.get(model)
        if url is None:
            raise
        print(f"    model tag {model!r} failed; retrying via release URL",
              flush=True)
        return make_calc(backend, dtype, model=url, device=device)


def free_cuda():
    """Release cached VRAM between cells (three model loads on a 12 GiB card)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_one(cell: str, calc, unitcell, supercell, displacement: float,
            npoints: int):
    """One (cell, displacement) measurement -> (summary record, band dict)."""
    t0 = time.perf_counter()
    phonon = build_phonon(unitcell, calc, supercell=supercell,
                          displacement=displacement,
                          log=lambda m: print(m, flush=True))
    band = band_path_SXGYS(phonon, npoints=npoints)
    min_freq = min_freq_near_gamma(phonon)
    freqs = band["frequencies_THz"]
    rec = {
        "cell": cell,
        "displacement": displacement,
        "min_freq_near_gamma_THz": min_freq,
        "n_imaginary_band_points": int((freqs < -IMAG_TOL_THZ).sum()),
        "max_freq_THz": float(freqs.max()),
        "wall_s": time.perf_counter() - t0,
    }
    return rec, band


def save_band(cell: str, displacement: float, band: dict, rec: dict,
              run_params: dict):
    tag = f"{cell.replace('/', '-')}_d{displacement:g}"
    npz_path = RAW_DIR / f"band_{tag}.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path,
             distances=band["distances"],
             frequencies_THz=band["frequencies_THz"],
             labels=np.array(band["labels"]),
             label_distances=band["label_distances"])
    print(f"[phosbench] wrote {npz_path}", flush=True)
    write_json(npz_path.with_suffix(".json"), {**rec, **run_params})


def branch_rmse_cm1(freqs_thz: np.ndarray, ref_thz: np.ndarray) -> dict:
    """Per-branch RMSE after sorting per q-point (branch labels can swap at
    crossings between cells; sorted comparison measures spectrum error, not
    labelling luck)."""
    a = np.sort(np.asarray(freqs_thz), axis=1)
    b = np.sort(np.asarray(ref_thz), axis=1)
    sq = (a - b) ** 2
    return {
        "per_branch_rmse_cm1": (np.sqrt(sq.mean(axis=0)) * THZ_TO_CM1).tolist(),
        "overall_rmse_cm1": float(np.sqrt(sq.mean()) * THZ_TO_CM1),
    }


def print_table(runs: list[dict], rmse: dict):
    header = (f"{'cell':<14} {'disp_A':>6} {'minF(G)THz':>11} {'n_imag':>6} "
              f"{'maxF_THz':>9} {'rmse_cm1':>9} {'wall_s':>7}")
    print("\n" + header)
    print("-" * len(header))
    for r in runs:
        if "error" in r:
            print(f"{r['cell']:<14} {r['displacement']:>6g} "
                  f"{'ERROR: ' + r['error'][:50]}")
            continue
        key = f"{r['cell']}@d{r['displacement']:g}"
        rmse_s = (f"{rmse[key]['overall_rmse_cm1']:9.3f}" if key in rmse
                  else f"{'ref' if r['cell'] == REFERENCE_CELL else '-':>9}")
        print(f"{r['cell']:<14} {r['displacement']:>6g} "
              f"{r['min_freq_near_gamma_THz']:>11.4f} "
              f"{r['n_imaginary_band_points']:>6d} {r['max_freq_THz']:>9.3f} "
              f"{rmse_s} {r['wall_s']:>7.1f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cells", default="e3nn/float64,e3nn/float32,cueq/float32")
    p.add_argument("--model", default=None,
                   help="default: configs/model_choice.json winner, else medium-omat-0 baseline")
    p.add_argument("--supercell", default=DEFAULT_SUPERCELL)
    p.add_argument("--displacements", default="0.01,0.03,0.05")
    p.add_argument("--npoints", type=int, default=51)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true",
                   help="single displacement 0.01, npoints 21, small supercell")
    args = p.parse_args()

    model = args.model or default_model()
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    supercell = tuple(int(x) for x in args.supercell.split(","))
    displacements = [float(x) for x in args.displacements.split(",")]
    npoints = args.npoints
    if args.smoke:
        displacements, npoints = [0.01], 21
        if args.supercell == DEFAULT_SUPERCELL:
            supercell = (2, 3, 1)  # under-converged on purpose: pipeline check
        print(f"[smoke] displacements={displacements} npoints={npoints} "
              f"supercell={supercell}", flush=True)

    unitcell = load_canonical()
    print(f"[phosbench] D1 phonons: model={model} cells={cells} "
          f"supercell={supercell} displacements={displacements}", flush=True)

    runs, bands = [], {}
    t_start = time.time()
    for cell in cells:
        backend, _, dtype = cell.partition("/")
        print(f"=== cell {cell} (elapsed {time.time() - t_start:.0f}s)",
              flush=True)
        try:
            calc = make_calc_robust(backend, dtype, model, args.device)
        except Exception as exc:
            # tolerated by design: cueq/float64 is broken upstream (MACE #1203)
            traceback.print_exc()
            runs += [{"cell": cell, "displacement": d,
                      "error": f"calculator: {exc!r}"} for d in displacements]
            continue
        for disp in displacements:
            print(f"--- {cell} @ d={disp} A", flush=True)
            try:
                rec, band = run_one(cell, calc, unitcell, supercell, disp,
                                    npoints)
            except Exception as exc:
                traceback.print_exc()
                runs.append({"cell": cell, "displacement": disp,
                             "error": repr(exc)})
                continue
            runs.append(rec)
            bands[(cell, disp)] = band["frequencies_THz"]
            save_band(cell, disp, band, rec, {
                "model": model, "supercell": list(supercell),
                "npoints": npoints, "device": args.device})
            print(f"    minF(near G)={rec['min_freq_near_gamma_THz']:.4f} THz  "
                  f"n_imag={rec['n_imaginary_band_points']}  "
                  f"maxF={rec['max_freq_THz']:.3f} THz  "
                  f"wall={rec['wall_s']:.1f}s", flush=True)
        del calc
        free_cuda()

    # per-branch RMSE vs the fp64 reference at the SAME displacement, so the
    # comparison isolates precision/kernel error from truncation error
    rmse = {}
    for (cell, disp), freqs in bands.items():
        ref = bands.get((REFERENCE_CELL, disp))
        if cell == REFERENCE_CELL or ref is None:
            continue
        rmse[f"{cell}@d{disp:g}"] = {
            "reference": REFERENCE_CELL, "displacement": disp,
            **branch_rmse_cm1(freqs, ref),
        }

    write_json(SUMMARY_PATH, {
        "stage": "D1_phonons",
        "model": model,
        "cells": cells,
        "reference_cell": REFERENCE_CELL,
        "supercell": list(supercell),
        "displacements": displacements,
        "npoints": npoints,
        "smoke": args.smoke,
        "imaginary_tolerance_THz": IMAG_TOL_THZ,
        "runs": runs,
        "rmse_vs_reference": rmse,
        "env": env_metadata(),
    })

    print_table(runs, rmse)
    n_failed = sum("error" in r for r in runs)
    print(f"\n[phosbench] D1 done: {len(runs) - n_failed}/{len(runs)} runs ok "
          f"in {time.time() - t_start:.0f}s", flush=True)
    if n_failed == len(runs):
        return 2
    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
