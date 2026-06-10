#!/usr/bin/env python
"""Stage A - physics gate + zero-shot foundation-model selection (e3nn/float64).

Throughput numbers are worthless if the model gets the material wrong, so
before any benchmark time is spent, each candidate foundation model must
reproduce known monolayer-phosphorene physics zero-shot on the reference
numerical path (e3nn backend, float64):

  GEOMETRY  relaxed lattice constants within 3% of literature PBE, every P
            atom still 3-fold coordinated, pucker preserved (z-spread in
            [1.8, 2.6] A). A model that flattens the sheet or breaks bonds
            has not learned this allotrope - disqualified in seconds.
  PHONONS   minimum frequency near Gamma above -0.3 THz on a quick 3x5x1
            finite-displacement run. The tolerance absorbs the numerical
            softness of the 2D flexural (ZA) branch at this supercell size;
            a genuinely unstable sheet is imaginary by THz, not 0.3.

Selection: passing both gates beats failing either; ties among passers break
by smaller |most imaginary frequency near Gamma|, then by worst lattice
deviation, then by --models order. The winner is written to
configs/model_choice.json and, if it differs from whatever built the canonical
structure, scripts/00_make_structure.py is re-run with the winner so every
downstream stage benchmarks the physically validated model.

If every candidate fails, the zero-shot failure is itself a reportable
finding; this script exits 1 and prints the pivot plan (GAP-20-style
fine-tune is the time-boxed stretch).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from phosbench.common import CANONICAL_XYZ, env_metadata, write_json

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent

GEOM_TOL_PCT = 3.0             # |da|, |db| vs literature PBE
PUCKER_RANGE_A = (1.8, 2.6)    # z-spread of the relaxed 4-atom cell
PHONON_TOL_THZ = -0.3          # min freq near Gamma; slack for ZA-branch numerics
PHONON_SUPERCELL = (3, 5, 1)   # quick-gate size; production phonons use (4, 6, 1)

# mace_mp tag resolution can lag new foundation releases; retry with the
# release asset directly (mace_mp accepts a URL in place of a tag).
FALLBACK_URLS = {
    "medium-mpa-0": "https://github.com/ACEsuit/mace-foundations/releases/"
                    "download/mace_mpa_0/mace-mpa-0-medium.model",
    "medium-omat-0": "https://github.com/ACEsuit/mace-foundations/releases/"
                     "download/mace_omat_0/mace-omat-0-medium.model",
}


def _load_structure_module():
    """Import scripts/00_make_structure.py despite its digit-leading filename."""
    path = SCRIPTS / "00_make_structure.py"
    spec = importlib.util.spec_from_file_location("phosbench_make_structure", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def relax_candidate(mk, tag, device, fmax):
    """build_and_relax with release-URL fallback; returns (atoms-with-calc, spec).

    The retry cannot distinguish 'unknown tag' from 'relaxation blew up', but a
    genuine relaxation failure simply fails again on the URL path and the
    exception propagates to the caller, which records it for that model.
    """
    try:
        atoms = mk.build_and_relax(model=tag, device=device, fmax=fmax, logfile=None)
        return atoms, tag
    except Exception as exc:
        url = FALLBACK_URLS.get(tag)
        if url is None:
            raise
        print(f"    tag {tag!r} failed ({exc!r}); retrying release URL", flush=True)
        atoms = mk.build_and_relax(model=url, device=device, fmax=fmax, logfile=None)
        return atoms, url


def geometry_record(mk, atoms) -> dict:
    a, b = (float(x) for x in atoms.cell.lengths()[:2])
    lit_a = mk.LITERATURE["a_armchair_A"]
    lit_b = mk.LITERATURE["b_zigzag_A"]
    rec = {
        "a_armchair_A": a,
        "b_zigzag_A": b,
        "da_pct": 100.0 * (a - lit_a) / lit_a,
        "db_pct": 100.0 * (b - lit_b) / lit_b,
        "pucker_A": float(np.ptp(atoms.get_positions()[:, 2])),
        "coordination_ok": mk.coordination_ok(atoms),
        "energy_per_atom_eV": float(atoms.get_potential_energy()) / len(atoms),
        "residual_fmax_eV_per_A": float(
            np.linalg.norm(atoms.get_forces(), axis=1).max()),
    }
    rec["geometry_pass"] = bool(
        abs(rec["da_pct"]) <= GEOM_TOL_PCT
        and abs(rec["db_pct"]) <= GEOM_TOL_PCT
        and rec["coordination_ok"]
        and PUCKER_RANGE_A[0] <= rec["pucker_A"] <= PUCKER_RANGE_A[1]
    )
    return rec


def phonon_record(atoms, calc, tag) -> dict:
    from phosbench.phonons import band_path_SXGYS, build_phonon, min_freq_near_gamma

    phonon = build_phonon(atoms, calc, supercell=PHONON_SUPERCELL,
                          displacement=0.01,
                          log=lambda *a: print("   ", *a, flush=True))
    min_freq = float(min_freq_near_gamma(phonon))

    # Band structure is saved for later inspection, not gated on: at this
    # supercell only near-Gamma stability is converged enough to trust.
    band = band_path_SXGYS(phonon, npoints=21)
    npz = REPO / "results" / "raw" / f"gate_phonons_{_slug(tag)}.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz,
             distances=band["distances"],
             frequencies_THz=band["frequencies_THz"],
             labels=np.array(band["labels"]),
             label_distances=band["label_distances"])
    print(f"    [phosbench] wrote {npz}", flush=True)
    return {
        "min_freq_near_gamma_THz": min_freq,
        "pass": bool(min_freq > PHONON_TOL_THZ),
        "band_npz": str(npz.relative_to(REPO)),
        "supercell": list(PHONON_SUPERCELL),
        "displacement_A": 0.01,
    }


def evaluate_model(mk, tag, device, fmax, smoke) -> dict:
    t0 = time.perf_counter()
    rec = {"tag": tag, "resolved_model": None, "phonon": None, "error": None}
    try:
        atoms, spec = relax_candidate(mk, tag, device, fmax)
        rec["resolved_model"] = spec
        rec.update(geometry_record(mk, atoms))
        print(f"    a={rec['a_armchair_A']:.4f} A ({rec['da_pct']:+.2f}%)  "
              f"b={rec['b_zigzag_A']:.4f} A ({rec['db_pct']:+.2f}%)  "
              f"pucker={rec['pucker_A']:.3f} A  "
              f"coord={'ok' if rec['coordination_ok'] else 'BROKEN'}  "
              f"E/atom={rec['energy_per_atom_eV']:.4f} eV", flush=True)
        if not rec["geometry_pass"]:
            print("    geometry gate: FAIL - phonon gate skipped", flush=True)
        elif smoke:
            print("    geometry gate: PASS (smoke: phonon gate skipped)", flush=True)
        else:
            print("    geometry gate: PASS - phonon gate on "
                  f"{'x'.join(map(str, PHONON_SUPERCELL))} supercell", flush=True)
            rec["phonon"] = phonon_record(atoms, atoms.calc, tag)
            print(f"    min freq near Gamma: "
                  f"{rec['phonon']['min_freq_near_gamma_THz']:+.3f} THz "
                  f"(tolerance {PHONON_TOL_THZ} THz) -> "
                  f"{'PASS' if rec['phonon']['pass'] else 'FAIL'}", flush=True)
        atoms.calc = None
    except Exception as exc:
        rec["error"] = repr(exc)
        rec["traceback_tail"] = traceback.format_exc()[-1500:]
        rec["geometry_pass"] = False
        print(f"    ERROR: {exc!r}", flush=True)
    _free_cuda(device)
    rec["passed"] = bool(rec["geometry_pass"]
                         and (smoke or (rec["phonon"] or {}).get("pass")))
    rec["wall_s"] = round(time.perf_counter() - t0, 1)
    return rec


def rank_passing(order, details) -> list[str]:
    """Passing models, best first: |min imaginary|, lattice deviation, CLI order."""
    def key(tag):
        d = details[tag]
        mf = (d["phonon"] or {}).get("min_freq_near_gamma_THz")
        imag = abs(min(mf, 0.0)) if mf is not None else 0.0
        geom = max(abs(d["da_pct"]), abs(d["db_pct"]))
        return (imag, geom, order.index(tag))

    return sorted((t for t in order if details[t]["passed"]), key=key)


def print_table(order, details):
    cols = (f"{'model':<18}{'a(A)':>9}{'da%':>8}{'b(A)':>9}{'db%':>8}"
            f"{'pucker(A)':>11}{'coord':>7}{'minf(THz)':>11}{'gate':>7}")
    rule = "-" * len(cols)
    print("\n" + rule + "\n" + cols + "\n" + rule)
    for tag in order:
        d = details[tag]
        coord = d.get("coordination_ok")
        mf = (d["phonon"] or {}).get("min_freq_near_gamma_THz")
        print(f"{tag:<18}"
              + _cell(d.get("a_armchair_A"), "{:9.4f}", 9)
              + _cell(d.get("da_pct"), "{:+8.2f}", 8)
              + _cell(d.get("b_zigzag_A"), "{:9.4f}", 9)
              + _cell(d.get("db_pct"), "{:+8.2f}", 8)
              + _cell(d.get("pucker_A"), "{:11.3f}", 11)
              + (f"{'yes' if coord else 'NO':>7}" if coord is not None else f"{'-':>7}")
              + _cell(mf, "{:+11.3f}", 11)
              + f"{'PASS' if d['passed'] else 'FAIL':>7}")
        if d["error"]:
            print(f"{'':18}  error: {d['error']}")
    print(rule, flush=True)


def _cell(v, fmt, width):
    return fmt.format(v) if isinstance(v, (int, float)) else f"{'-':>{width}}"


def _slug(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tag)


def _free_cuda(device):
    if device == "cuda":
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()


def canonical_model() -> str | None:
    """Model recorded in the sidecar JSON of the canonical structure, if any."""
    try:
        side = CANONICAL_XYZ.with_suffix(".json")
        return json.loads(side.read_text())["relaxation"]["model"]
    except Exception:
        return None


def rebuild_canonical(spec, device, fmax) -> int:
    """Re-run 00_make_structure.py in a subprocess: identical save logic, and a
    fresh CUDA context so this script's state cannot leak into the canonical."""
    cmd = [sys.executable, str(SCRIPTS / "00_make_structure.py"),
           "--model", spec, "--device", device, "--fmax", str(fmax)]
    print(f"[phosbench] rebuilding canonical: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", default="medium-mpa-0,medium-omat-0,medium",
                   help="comma list of mace_mp tags, in priority order")
    p.add_argument("--device", default="cuda")
    p.add_argument("--fmax", type=float, default=None,
                   help="relaxation tolerance (default 1e-3; 5e-2 under --smoke)")
    p.add_argument("--smoke", action="store_true",
                   help="geometry gate only with loose fmax - <2 min sanity run; "
                        "writes model_choice.smoke.json, never touches canonical")
    args = p.parse_args()
    fmax = args.fmax if args.fmax is not None else (5e-2 if args.smoke else 1e-3)

    mk = _load_structure_module()
    order = [t.strip() for t in args.models.split(",") if t.strip()]
    if not order:
        print("no models requested")
        return 1

    details = {}
    for i, tag in enumerate(order, 1):
        print(f"=== [{i}/{len(order)}] candidate {tag} "
              f"(e3nn/float64, fmax={fmax:g})", flush=True)
        details[tag] = evaluate_model(mk, tag, args.device, fmax, args.smoke)

    ranked = rank_passing(order, details)
    winner = ranked[0] if ranked else None
    print_table(order, details)

    out_path = REPO / "configs" / ("model_choice.smoke.json" if args.smoke
                                   else "model_choice.json")
    payload = {
        "model": winner,
        "passed": {t: details[t]["passed"] for t in order},
        "details": details,
        "ranking": ranked,
        "smoke": args.smoke,
        "gates": {"geom_tol_pct": GEOM_TOL_PCT,
                  "pucker_range_A": list(PUCKER_RANGE_A),
                  "phonon_tol_THz": PHONON_TOL_THZ,
                  "fmax": fmax,
                  "literature": mk.LITERATURE},
        "env": env_metadata(),
    }
    write_json(out_path, payload)

    if winner is None:
        print("PHYSICS_GATE: ALL MODELS FAILED")
        print("PIVOT: zero-shot validation failure is itself a finding. Options:")
        print("  1. Report Stage A as a negative result: which physics each model")
        print("     misses is in configs/model_choice.json and the npz band files.")
        print("  2. Time-boxed stretch: GAP-20-style fine-tune on a small DFT set")
        print("     of phosphorene configurations, then re-run this gate on it.")
        print("  3. Before pivoting, sanity-check the gate itself: inspect")
        print("     results/raw/gate_phonons_*.npz and rerun with tighter --fmax.")
        return 1

    spec = details[winner]["resolved_model"]
    print(f"PHYSICS_GATE: WINNER {winner}"
          + (f" (resolved: {spec})" if spec != winner else ""))

    built_with = canonical_model()
    if args.smoke:
        print("[phosbench] smoke run: canonical structure left untouched", flush=True)
    elif CANONICAL_XYZ.exists() and built_with in (winner, spec):
        print(f"[phosbench] canonical already built with {built_with} - keeping it")
    else:
        rc = rebuild_canonical(spec, args.device, fmax)
        payload["canonical_rebuild"] = {"model": spec, "returncode": rc}
        write_json(out_path, payload)
        if rc != 0:
            print(f"WARNING: canonical rebuild exited {rc} - "
                  "inspect structures/ before running Stage B", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
