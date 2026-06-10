#!/usr/bin/env python
"""Stage D3+D4 - MD stability per cell: NVE energy drift, NPT in-plane lattice.

D3 (NVE): symplectic VelocityVerlet conserves total energy up to force errors,
so the linear drift of Etot/atom over tens of ps is the cleanest single number
for "is this backend/dtype safe for production MD" - fp32 force noise shows up
here long before any static test catches it. Reported in microeV/atom/ps
(at a 1 fs timestep that is also microeV/atom per 1000 steps).

D4 (NPT): Inhomogeneous_NPTBerendsen with mask=(1,1,0) - the two in-plane cell
lengths breathe independently because phosphorene is strongly anisotropic,
while z stays fixed: it is vacuum, not a lattice direction. The 300 K mean
lattice vs the canonical 0 K one gives thermal expansion plus any
backend-induced bias; the per-cell spread across configs is the deployment
signal.

Barostat unit handling (the part everyone gets wrong): ASE's NPTBerendsen
takes its `_au`-suffixed arguments in ASE-native units - pressure in eV/A^3,
compressibility in (eV/A^3)^-1 - NOT bar and NOT Hartree atomic units.
ase.units.bar (~6.2415e-7) is 1 bar expressed in eV/A^3, so a textbook
liquid-like compressibility kappa ~ 4.6e-5 /bar converts as

    compressibility_au = 4.6e-5 / units.bar  ~  74 (eV/A^3)^-1

kappa only scales the barostat relaxation rate (cell rescaling per step is
proportional to kappa/taup * (p - p_target)), never the equilibrium lattice,
so an order-of-magnitude value is sufficient; the target pressure_au=0.0 is
unit-safe by construction.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from phosbench.common import (env_metadata, load_canonical, make_calc,
                              make_supercell, write_json)

REPO = Path(__file__).resolve().parent.parent
MD_DIR = REPO / "results" / "raw" / "md"

SEED = 0                    # fixed velocity seed: cells differ only in numerics
KAPPA_PER_BAR = 4.6e-5      # order-of-magnitude compressibility, see docstring
TAUT_FS = 100.0
TAUP_FS = 1000.0
SUPERCELL_DEFAULT = "8,16"  # 512 atoms

MODEL_URLS = {
    # known release assets, used when the short tag is not yet known to the
    # installed mace-torch (tags newer than the pinned 0.3.16 can 404)
    "medium-mpa-0": "https://github.com/ACEsuit/mace-foundations/releases/"
                    "download/mace_mpa_0/mace-mpa-0-medium.model",
    "medium-omat-0": "https://github.com/ACEsuit/mace-foundations/releases/"
                     "download/mace_omat_0/mace-omat-0-medium.model",
}


def default_model() -> str:
    """Stage-A winner from configs/model_choice.json if present, else mpa-0."""
    try:
        payload = json.loads((REPO / "configs" / "model_choice.json").read_text())
        model = payload if isinstance(payload, str) else payload.get("model")
        if isinstance(model, str) and model:
            return model
    except (OSError, ValueError):
        pass
    return "medium-mpa-0"


def make_calc_resolved(backend, dtype, model, device):
    """make_calc, falling back to the pinned release URL if the tag fails."""
    try:
        return make_calc(backend, dtype, model=model, device=device), model
    except Exception as exc:
        url = MODEL_URLS.get(model)
        if url is None:
            raise
        print(f"[phosbench] model tag {model!r} failed ({exc!r}); "
              f"retrying release URL", flush=True)
        return make_calc(backend, dtype, model=url, device=device), url


def n_steps_for(ps: float, timestep_fs: float) -> int:
    return max(1, int(round(ps * 1000.0 / timestep_fs)))


def _drive(dyn, n_steps, log_every, sample, label):
    """Run dyn in log_every chunks, sampling after each; ~10 progress lines."""
    sample(0)  # also absorbs first-call JIT/autotune outside the timed loop
    report_every = max(n_steps // 10, log_every)
    t0 = time.perf_counter()
    done, next_report = 0, report_every
    while done < n_steps:
        chunk = min(log_every, n_steps - done)
        dyn.run(chunk)
        done += chunk
        sample(done)
        if done >= next_report or done == n_steps:
            rate = done / (time.perf_counter() - t0)
            print(f"      {label} {done}/{n_steps} steps "
                  f"({rate:.1f} steps/s)", flush=True)
            next_report += report_every
    return time.perf_counter() - t0


def run_nve(atoms, timestep_fs, n_steps, log_every, temperature_K):
    from ase import units
    from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                             Stationary)
    from ase.md.verlet import VelocityVerlet

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K,
                                 rng=np.random.default_rng(SEED))
    Stationary(atoms)
    dyn = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)
    n = len(atoms)
    t_ps, etot, ekin, temp = [], [], [], []

    def sample(step):
        ek = atoms.get_kinetic_energy() / n
        t_ps.append(step * timestep_fs * 1e-3)
        etot.append(atoms.get_potential_energy() / n + ek)
        ekin.append(ek)
        temp.append(atoms.get_temperature())

    wall = _drive(dyn, n_steps, log_every, sample, "NVE")
    t, e = np.asarray(t_ps), np.asarray(etot)
    drift = float(np.polyfit(t, e, 1)[0]) * 1e6     # eV/atom/ps -> microeV
    half = len(t) // 2
    traces = {"time_ps": t, "etot_per_atom_eV": e,
              "ekin_per_atom_eV": np.asarray(ekin),
              "temperature_K": np.asarray(temp)}
    summary = {"drift_ueV_per_atom_ps": drift,
               "etot_std_ueV_per_atom": float(e.std() * 1e6),
               "temp_mean_K": float(np.mean(traces["temperature_K"][half:])),
               "n_steps": n_steps, "wall_s": wall,
               "steps_per_s": n_steps / wall}
    return summary, traces


def run_npt(atoms, nx, ny, timestep_fs, n_steps, log_every, temperature_K):
    from ase import units
    from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen
    from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                             Stationary)

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K,
                                 rng=np.random.default_rng(SEED + 1))
    Stationary(atoms)
    dyn = Inhomogeneous_NPTBerendsen(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_K,
        pressure_au=0.0,                                # zero in-plane stress
        taut=TAUT_FS * units.fs,
        taup=TAUP_FS * units.fs,
        compressibility_au=KAPPA_PER_BAR / units.bar,   # see module docstring
        mask=(1, 1, 0),                                 # z is vacuum: fixed
    )
    t_ps, a_l, b_l, temp = [], [], [], []

    def sample(step):
        t_ps.append(step * timestep_fs * 1e-3)
        a_l.append(float(atoms.cell[0, 0]))
        b_l.append(float(atoms.cell[1, 1]))
        temp.append(atoms.get_temperature())

    wall = _drive(dyn, n_steps, log_every, sample, "NPT")
    a, b = np.asarray(a_l), np.asarray(b_l)
    half = len(a) // 2                                  # discard equilibration
    traces = {"time_ps": np.asarray(t_ps), "a_super_A": a, "b_super_A": b,
              "temperature_K": np.asarray(temp)}
    summary = {
        # per primitive cell, directly comparable to the canonical 0 K lattice
        "a_mean_A": float(a[half:].mean() / nx),
        "b_mean_A": float(b[half:].mean() / ny),
        "a_std_A": float(a[half:].std() / nx),
        "b_std_A": float(b[half:].std() / ny),
        "a_super_mean_A": float(a[half:].mean()),
        "b_super_mean_A": float(b[half:].mean()),
        "temp_mean_K": float(np.mean(traces["temperature_K"][half:])),
        "stats_window": "second half",
        "n_steps": n_steps, "wall_s": wall,
        "steps_per_s": n_steps / wall,
    }
    return summary, traces


def run_cell(backend, dtype, model, device, nx, ny, args):
    calc, resolved = make_calc_resolved(backend, dtype, model, device)
    tag = f"{backend}-{dtype}"
    t0 = time.perf_counter()

    nve_steps = n_steps_for(args.nve_ps, args.timestep_fs)
    print(f"    NVE: {nve_steps} steps ({args.nve_ps} ps), "
          f"{4 * nx * ny} atoms", flush=True)
    atoms = make_supercell(nx, ny)
    atoms.calc = calc
    nve, nve_tr = run_nve(atoms, args.timestep_fs, nve_steps,
                          args.log_every, args.temperature)
    nve_npz = MD_DIR / f"{tag}_nve.npz"
    np.savez(nve_npz, nx=nx, ny=ny, **nve_tr)
    nve["trace_npz"] = str(nve_npz.relative_to(REPO))
    print(f"    NVE drift = {nve['drift_ueV_per_atom_ps']:+.3f} ueV/atom/ps  "
          f"T={nve['temp_mean_K']:.0f} K", flush=True)

    npt_steps = n_steps_for(args.npt_ps, args.timestep_fs)
    print(f"    NPT: {npt_steps} steps ({args.npt_ps} ps)", flush=True)
    atoms = make_supercell(nx, ny)      # fresh start: NPT is independent of NVE
    atoms.calc = calc
    npt, npt_tr = run_npt(atoms, nx, ny, args.timestep_fs, npt_steps,
                          args.log_every, args.temperature)
    npt_npz = MD_DIR / f"{tag}_npt.npz"
    np.savez(npt_npz, nx=nx, ny=ny, **npt_tr)
    npt["trace_npz"] = str(npt_npz.relative_to(REPO))
    print(f"    NPT a={npt['a_mean_A']:.4f}+/-{npt['a_std_A']:.4f} A  "
          f"b={npt['b_mean_A']:.4f}+/-{npt['b_std_A']:.4f} A "
          f"(per unit cell, 2nd half)", flush=True)

    return {"ok": True, "model_resolved": resolved, "natoms": 4 * nx * ny,
            "supercell": [nx, ny, 1], "nve": nve, "npt": npt,
            "wall_s_total": time.perf_counter() - t0}


def _print_summary(out):
    print("\n== md stability summary ==")
    print(f"{'cell':<16}{'drift ueV/at/ps':>16}{'a (A/uc)':>16}"
          f"{'b (A/uc)':>16}{'NVE st/s':>10}{'NPT st/s':>10}")
    for key, c in out["cells"].items():
        if not c.get("ok"):
            print(f"{key:<16}FAILED  {c.get('error', '')[:70]}")
            continue
        nve, npt = c["nve"], c["npt"]
        a = f"{npt['a_mean_A']:.4f}+/-{npt['a_std_A']:.4f}"
        b = f"{npt['b_mean_A']:.4f}+/-{npt['b_std_A']:.4f}"
        print(f"{key:<16}{nve['drift_ueV_per_atom_ps']:>+16.3f}{a:>16}{b:>16}"
              f"{nve['steps_per_s']:>10.1f}{npt['steps_per_s']:>10.1f}")
    print(f"{'canonical 0K':<16}{'-':>16}"
          f"{out['canonical_a_A']:>16.4f}{out['canonical_b_A']:>16.4f}",
          flush=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Stage D3+D4 - NVE drift and NPT lattice per cell config")
    p.add_argument("--cells", default="e3nn/float64,e3nn/float32,cueq/float32")
    p.add_argument("--model", default=default_model())
    p.add_argument("--supercell", default=SUPERCELL_DEFAULT,
                   help="nx,ny in-plane repeats (8,16 = 512 atoms)")
    p.add_argument("--nve-ps", type=float, default=25.0)
    p.add_argument("--npt-ps", type=float, default=50.0)
    p.add_argument("--timestep-fs", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--log-every", type=int, default=10,
                   help="sample traces every N MD steps")
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true",
                   help="0.5 ps phases on a 2x3 cell - <2 min sanity run")
    args = p.parse_args()

    if args.smoke:
        args.nve_ps = args.npt_ps = 0.5
        if args.supercell == SUPERCELL_DEFAULT:   # honor an explicit override
            args.supercell = "2,3"
    nx, ny = (int(v) for v in args.supercell.split(","))

    canon = load_canonical()
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    MD_DIR.mkdir(parents=True, exist_ok=True)
    out = {"model": args.model, "device": args.device,
           "supercell": [nx, ny, 1], "natoms": 4 * nx * ny,
           "nve_ps": args.nve_ps, "npt_ps": args.npt_ps,
           "timestep_fs": args.timestep_fs, "temperature_K": args.temperature,
           "log_every": args.log_every, "seed": SEED, "smoke": args.smoke,
           "barostat": {"taut_fs": TAUT_FS, "taup_fs": TAUP_FS,
                        "compressibility_per_bar": KAPPA_PER_BAR,
                        "pressure_au": 0.0, "mask": [1, 1, 0]},
           "canonical_a_A": float(canon.cell[0, 0]),
           "canonical_b_A": float(canon.cell[1, 1]),
           "cells": {}, "env": env_metadata()}

    t0 = time.time()
    for key in cells:
        print(f"--- cell {key} (model={args.model})", flush=True)
        try:
            parts = key.split("/")
            if len(parts) != 2:
                raise ValueError(f"cell must be 'backend/dtype', got {key!r}")
            out["cells"][key] = run_cell(*parts, args.model, args.device,
                                         nx, ny, args)
        except Exception as exc:
            # e.g. cueq/float64 (broken upstream, MACE #1203/#1298) - record,
            # keep the other cells alive
            out["cells"][key] = {"ok": False, "error": repr(exc),
                                 "traceback_tail": traceback.format_exc()[-1500:]}
            print(f"    FAILED: {exc!r}", flush=True)

    write_json(REPO / "results/raw/md_stability.json", out)
    _print_summary(out)
    n_ok = sum(1 for c in out["cells"].values() if c.get("ok"))
    print(f"[phosbench] md stability done in {time.time() - t0:.0f}s "
          f"({n_ok}/{len(cells)} cells ok)")
    return 0 if n_ok == len(cells) else (1 if n_ok else 2)


if __name__ == "__main__":
    raise SystemExit(main())
