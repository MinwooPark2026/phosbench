#!/usr/bin/env python
"""Stretch - download GAP-20 phosphorus DFT data and convert for MACE fine-tuning.

GAP-20 (Deringer et al., Nat. Commun. 2020; Zenodo 10.5281/zenodo.4003703) is
the only public general-purpose phosphorus DFT dataset. We fine-tune the
foundation model on it to repair the 7-10 % armchair-axis compression that the
zero-shot physics gate caught (scripts/02_gate_physics.py).

Steps: download fitting xyz -> inspect keys/config_types -> write
data/gap20_train.xyz with MACE-conventional REF_energy / REF_forces fields
(energies/forces only - virials are deliberately dropped, see the slab-stress
finding) -> report E0 candidates (isolated-atom configs) and counts.
"""

import sys
import urllib.request
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
URL = ("https://zenodo.org/api/records/4003703/files/"
       "P_GAP_20_fitting_data.xyz/content")
RAW_XYZ = DATA / "P_GAP_20_fitting_data.xyz"
OUT_XYZ = DATA / "gap20_train.xyz"


def main() -> int:
    DATA.mkdir(exist_ok=True)
    if not RAW_XYZ.exists():
        print(f"[50_prep] downloading {URL}", flush=True)
        urllib.request.urlretrieve(URL, RAW_XYZ)
    print(f"[50_prep] raw file: {RAW_XYZ.stat().st_size/1e6:.1f} MB")

    from ase.io import read, write

    frames = read(RAW_XYZ, index=":")
    print(f"[50_prep] {len(frames)} structures")

    types = Counter(f.info.get("config_type", "?") for f in frames)
    print("[50_prep] config_type counts:")
    for t, n in types.most_common():
        print(f"    {n:5d}  {t}")

    info_keys = Counter(k for f in frames[:200] for k in f.info)
    array_keys = Counter(k for f in frames[:200] for k in f.arrays)
    print(f"[50_prep] info keys (first 200): {dict(info_keys)}")
    print(f"[50_prep] array keys (first 200): {dict(array_keys)}")

    def get_energy(f):
        for k in ("energy", "dft_energy", "REF_energy"):
            if k in f.info:
                return float(f.info[k])
        if f.calc is not None and "energy" in f.calc.results:
            return float(f.calc.results["energy"])
        return None

    def get_forces(f):
        for k in ("forces", "force", "dft_force", "REF_forces"):
            if k in f.arrays:
                return f.arrays[k]
        if f.calc is not None and "forces" in f.calc.results:
            return f.calc.results["forces"]
        return None

    kept, skipped, e0 = [], 0, {}
    for f in frames:
        e, frc = get_energy(f), get_forces(f)
        if e is None or frc is None:
            skipped += 1
            continue
        g = f.copy()
        g.calc = None
        g.info = {"REF_energy": e,
                  "config_type": f.info.get("config_type", "bulk")}
        g.arrays = {k: v for k, v in g.arrays.items()
                    if k in ("numbers", "positions")}
        g.new_array("REF_forces", frc)
        if len(g) == 1 and "isolated" in g.info["config_type"].lower():
            e0[g.get_chemical_symbols()[0]] = e
            g.info["config_type"] = "IsolatedAtom"
        kept.append(g)
    write(OUT_XYZ, kept)
    print(f"[50_prep] wrote {OUT_XYZ}: {len(kept)} kept, {skipped} skipped")
    print(f"[50_prep] isolated-atom E0 candidates: {e0}")
    natoms = sorted(len(f) for f in kept)
    print(f"[50_prep] natoms min/median/max: "
          f"{natoms[0]}/{natoms[len(natoms)//2]}/{natoms[-1]}")
    print("PREP_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
