"""Shared infrastructure for phosbench: calculator factory, structures, timing, metadata.

All benchmark and accuracy scripts go through these helpers so that every result
JSON carries identical environment metadata and every configuration is constructed
the same way. Conventions:

  backend : 'e3nn' (reference implementation) | 'cueq' (cuEquivariance kernels)
  dtype   : 'float64' (reference) | 'float32'
  model   : name accepted by mace_mp() ('small'|'medium'|'large' or a newer tag/path)
  device  : 'cuda' | 'cpu'
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_XYZ = REPO_ROOT / "structures" / "phosphorene_relaxed.extxyz"


# --------------------------------------------------------------------------- #
# Calculator
# --------------------------------------------------------------------------- #

def make_calc(backend: str, dtype: str, model: str = "medium", device: str = "cuda"):
    """Build a MACE foundation-model ASE calculator for one benchmark config."""
    from mace.calculators import mace_mp

    if backend not in ("e3nn", "cueq"):
        raise ValueError(f"unknown backend {backend!r}")
    if dtype not in ("float32", "float64"):
        raise ValueError(f"unknown dtype {dtype!r}")
    return mace_mp(
        model=model,
        device=device,
        default_dtype=dtype,
        enable_cueq=(backend == "cueq"),
        dispersion=False,
    )


# --------------------------------------------------------------------------- #
# Structures
# --------------------------------------------------------------------------- #

def phosphorene_unit_cell(vacuum: float = 20.0):
    """Approximate 4-atom monolayer phosphorene cell (x = armchair, y = zigzag).

    Coordinates are deliberately approximate (~0.1 A): scripts/00_make_structure.py
    relaxes this with the reference model (e3nn, float64) and the *relaxed* file in
    structures/ is the canonical input for every other script. Never benchmark from
    the raw cell below.
    """
    from ase import Atoms

    a, b = 4.62, 3.30          # armchair, zigzag lattice constants (PBE-ish)
    dz = 2.14                  # pucker height
    z0 = vacuum / 2.0
    positions = [
        (0.000 * a, 0.0, z0),
        (0.367 * a, b / 2, z0),
        (0.500 * a, b / 2, z0 + dz),
        (0.867 * a, 0.0, z0 + dz),
    ]
    return Atoms(
        "P4",
        positions=positions,
        cell=[a, b, vacuum + dz],
        pbc=[True, True, False],
    )


def load_canonical(path: Path | str = CANONICAL_XYZ):
    from ase.io import read

    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} missing - run scripts/00_make_structure.py first"
        )
    return read(path)


def make_supercell(nx: int, ny: int, base=None):
    """nx x ny in-plane supercell of the canonical relaxed monolayer."""
    atoms = (base if base is not None else load_canonical()).copy()
    return atoms.repeat((nx, ny, 1))


def size_ladder(max_atoms: int = 40000):
    """(nx, ny, natoms) grid used by sweep scripts: ~64 atoms to the OOM boundary.

    Aspect ratios are kept near-square in Cartesian terms (cell is 4.62 x 3.30 A,
    so ny grows ~1.4x faster than nx).
    """
    ladder = []
    for nx, ny in [(4, 4), (5, 7), (8, 11), (11, 16), (16, 22), (23, 32),
                   (32, 45), (45, 64), (64, 90)]:
        natoms = 4 * nx * ny
        if natoms <= max_atoms:
            ladder.append((nx, ny, natoms))
    return ladder


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #

@dataclass
class TimingResult:
    config: dict
    natoms: int
    mode: str                       # 'force_call' | 'md'
    n_warmup: int
    n_steps: int
    wall_s: float
    ms_per_step: float
    ms_per_step_median: float
    ms_per_step_p10: float
    ms_per_step_p90: float
    us_per_atom_step: float
    ns_per_day_1fs: float           # MD throughput at 1 fs timestep
    peak_vram_mib: float | None
    error: str | None = None
    meta: dict = field(default_factory=dict)


def _sync(device: str):
    if device == "cuda":
        import torch

        torch.cuda.synchronize()


class _nvtx:
    """NVTX range context manager (no-op on CPU / when CUDA is absent)."""

    def __init__(self, label: str, device: str = "cuda"):
        self.label, self.on = label, device == "cuda"

    def __enter__(self):
        if self.on:
            import torch

            torch.cuda.nvtx.range_push(self.label)

    def __exit__(self, *exc):
        if self.on:
            import torch

            torch.cuda.nvtx.range_pop()


def gpu_clock_temp() -> dict | None:
    """One-shot SM clock / temperature / power sample via nvidia-smi.

    Consumer boost clocks drift with temperature over hours-long sweeps and we
    cannot lock clocks without sudo, so every measurement records its thermal
    state instead - disclosed in the methodology section.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        sm, temp, power, util = [x.strip() for x in out.stdout.strip().split(",")]
        return {"sm_clock_mhz": float(sm), "temp_C": float(temp),
                "power_W": float(power), "util_pct": float(util)}
    except Exception:
        return None


def time_force_calls(atoms, calc, device: str, n_warmup: int = 10,
                     n_steps: int = 50, seed: int = 0) -> dict:
    """Time bare force evaluations on slightly perturbed copies.

    Positions are rattled before every call so ASE never serves a cached result;
    this isolates calculator cost from MD-integrator overhead (compare with the
    'md' mode to quantify the harness overhead itself).
    """
    rng = np.random.default_rng(seed)
    base = atoms.get_positions()
    atoms.calc = calc

    def one_call():
        atoms.set_positions(base + rng.normal(scale=0.01, size=base.shape))
        with _nvtx("force_eval", device):
            atoms.get_forces()

    for _ in range(n_warmup):
        one_call()
    _sync(device)

    clocks0 = gpu_clock_temp() if device == "cuda" else None
    _reset_vram(device)
    laps = []
    t0 = time.perf_counter()
    for _ in range(n_steps):
        t = time.perf_counter()
        one_call()
        _sync(device)
        laps.append(time.perf_counter() - t)
    wall = time.perf_counter() - t0
    out = _stats(laps, wall, len(atoms), device)
    out["clocks_start"], out["clocks_end"] = clocks0, (
        gpu_clock_temp() if device == "cuda" else None)
    return out


def time_md(atoms, calc, device: str, n_warmup: int = 10, n_steps: int = 100,
            timestep_fs: float = 1.0, temperature_K: float = 300.0,
            seed: int = 0) -> dict:
    """Time a real NVE MD loop (VelocityVerlet) - the production-relevant number."""
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet
    from ase import units

    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=np.random.default_rng(seed))
    dyn = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)

    dyn.run(n_warmup)
    _sync(device)

    clocks0 = gpu_clock_temp() if device == "cuda" else None
    _reset_vram(device)
    laps = []
    t0 = time.perf_counter()
    for _ in range(n_steps):
        t = time.perf_counter()
        with _nvtx("md_step", device):
            dyn.run(1)
        _sync(device)
        laps.append(time.perf_counter() - t)
    wall = time.perf_counter() - t0
    out = _stats(laps, wall, len(atoms), device)
    out["ns_per_day_1fs"] = timestep_fs * 1e-6 * 86400.0 / (wall / n_steps)
    out["clocks_start"], out["clocks_end"] = clocks0, (
        gpu_clock_temp() if device == "cuda" else None)
    return out


def _reset_vram(device: str):
    if device == "cuda":
        import torch

        torch.cuda.reset_peak_memory_stats()


def _stats(laps, wall, natoms, device) -> dict:
    laps_ms = 1e3 * np.asarray(laps)
    n = len(laps)
    peak = None
    if device == "cuda":
        import torch

        peak = torch.cuda.max_memory_allocated() / 2**20
    per_step = wall / n
    return {
        "n_steps": n,
        "wall_s": wall,
        "ms_per_step": 1e3 * per_step,
        "ms_per_step_median": float(np.median(laps_ms)),
        "ms_per_step_p10": float(np.percentile(laps_ms, 10)),
        "ms_per_step_p90": float(np.percentile(laps_ms, 90)),
        "us_per_atom_step": 1e6 * per_step / natoms,
        "ns_per_day_1fs": 1e-6 * 86400.0 / per_step,
        "peak_vram_mib": peak,
    }


# --------------------------------------------------------------------------- #
# Metadata / IO
# --------------------------------------------------------------------------- #

def env_metadata() -> dict:
    import torch

    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    for pkg in ("mace-torch", "cuequivariance-torch", "e3nn", "ase", "phonopy"):
        meta[pkg] = _pkg_version(pkg)
    if torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["driver"] = _nvidia_driver()
    return meta


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _nvidia_driver() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def write_json(path: Path | str, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    print(f"[phosbench] wrote {path}")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
