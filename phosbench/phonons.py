"""Phonopy plumbing for phosbench: finite-displacement phonons via an ASE calculator.

This is the only module that talks to phonopy. Two reasons to centralise it:
(1) every consumer must use the SAME q-space conventions as the canonical cell
(x = armchair, y = zigzag, z = vacuum, so q_z = 0 everywhere and the supercell
c-repeat is pinned to 1), and (2) model load is expensive (~15 s + cueq autotune)
while displaced supercells are many, so force evaluation must reuse one live
calculator instance across fresh Atoms objects.

Frequencies are THz throughout; negative values are phonopy's convention for
imaginary modes. 1 THz = 33.356 cm^-1 (THZ_TO_CM1).
"""

from __future__ import annotations

import numpy as np

THZ_TO_CM1 = 33.356

#: Special points in fractional coordinates of the canonical 4-atom cell.
#: x = armchair -> X = (1/2, 0, 0); y = zigzag -> Y = (0, 1/2, 0). Swapping the
#: axis convention silently relabels the soft ZA physics, hence module constants.
SPECIAL_POINTS = {
    "S": (0.5, 0.5, 0.0),
    "X": (0.5, 0.0, 0.0),
    "G": (0.0, 0.0, 0.0),
    "Y": (0.0, 0.5, 0.0),
}
BAND_LABELS = ["S", "X", "G", "Y", "S"]


# --------------------------------------------------------------------------- #
# PhonopyAtoms <-> ase.Atoms
# --------------------------------------------------------------------------- #

def _to_phonopy(atoms):
    from phonopy.structure.atoms import PhonopyAtoms

    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=np.asarray(atoms.get_cell()),
        scaled_positions=atoms.get_scaled_positions(),
    )


def _to_ase(ph_atoms):
    """PhonopyAtoms -> ase.Atoms with the repo's pbc convention.

    Phonopy treats supercells as 3D-periodic; physically the c axis is ~22 A of
    vacuum, so pbc=[T,T,F] keeps force calls consistent with every other script
    (the vacuum gap exceeds twice the model cutoff either way).
    """
    from ase import Atoms

    return Atoms(
        symbols=ph_atoms.symbols,
        cell=np.asarray(ph_atoms.cell),
        scaled_positions=ph_atoms.scaled_positions,
        pbc=[True, True, False],
    )


# --------------------------------------------------------------------------- #
# Public API (PHONON INTERFACE CONTRACT)
# --------------------------------------------------------------------------- #

def build_phonon(unitcell_atoms, calc, supercell=(4, 6, 1), displacement=0.01,
                 log=print):
    """Finite-displacement Phonopy object with force constants from `calc`.

    Each displaced supercell gets a FRESH ase.Atoms but the SAME calculator
    instance: rebuilding the calculator per displacement would re-pay model
    load and cueq autotune dozens of times for identical numbers.
    """
    from phonopy import Phonopy

    if supercell[2] != 1:
        raise ValueError("c axis is vacuum: supercell c-repeat must stay 1, "
                         f"got {tuple(supercell)}")

    # primitive_matrix MUST stay identity: phonopy 4.x defaults to 'auto',
    # which re-standardises the cell (axis permutation for this slab) and would
    # silently re-interpret every q-point of band_path_SXGYS in the wrong basis.
    phonon = Phonopy(_to_phonopy(unitcell_atoms),
                     supercell_matrix=np.diag(supercell),
                     primitive_matrix="P")
    phonon.generate_displacements(distance=displacement)
    supercells = phonon.supercells_with_displacements
    log(f"[phonons] {len(supercells)} displaced supercells "
        f"({supercell[0]}x{supercell[1]}x1, {len(supercells[0])} atoms, "
        f"d={displacement} A)")

    force_sets = []
    for i, sc in enumerate(supercells):
        atoms = _to_ase(sc)
        atoms.calc = calc
        force_sets.append(np.asarray(atoms.get_forces(), dtype=np.float64))
        log(f"[phonons]   forces {i + 1}/{len(supercells)}")
    phonon.forces = np.asarray(force_sets)
    phonon.produce_force_constants()
    return phonon


def band_path_SXGYS(phonon, npoints=51):
    """Dispersion along S-X-Gamma-Y-S of the canonical cell.

    Returns {'distances': (nq,), 'frequencies_THz': (nq, nbranch),
             'labels': BAND_LABELS, 'label_distances': (5,)} with the four
    connected segments concatenated (segment endpoints are duplicated, which
    keeps per-q comparisons between cells trivially aligned).
    """
    from phonopy.phonon.band_structure import (
        get_band_qpoints_and_path_connections)

    qpoints, connections = get_band_qpoints_and_path_connections(
        [[SPECIAL_POINTS[lbl] for lbl in BAND_LABELS]], npoints=npoints)
    phonon.run_band_structure(qpoints, path_connections=connections,
                              labels=list(BAND_LABELS), with_eigenvectors=False)
    band = phonon.get_band_structure_dict()
    distances = np.concatenate(band["distances"])
    label_distances = np.array(
        [band["distances"][0][0]] + [seg[-1] for seg in band["distances"]])
    return {
        "distances": distances,
        "frequencies_THz": np.concatenate(band["frequencies"], axis=0),
        "labels": list(BAND_LABELS),
        "label_distances": label_distances,
    }


def min_freq_near_gamma(phonon, radius=0.05, mesh=(8, 12, 1)):
    """Minimum frequency (THz, negative = imaginary) for |q_frac| < radius.

    The watchdog for long-wavelength instability: a soft ZA branch shows up as
    negative frequencies near Gamma long before the band-path eye test. The
    three exact-Gamma acoustic modes are excluded only within 1e-3 THz - any
    larger residual there is a real acoustic-sum-rule violation of the force
    constants and SHOULD be reported, not masked.
    """
    # is_gamma_center=True: phonopy's default Monkhorst-Pack shift excludes
    # Gamma for even mesh numbers, leaving NO q-point inside the default radius.
    phonon.run_mesh(list(mesh), with_eigenvectors=False, is_gamma_center=True)
    mesh_dict = phonon.get_mesh_dict()
    qpts = np.asarray(mesh_dict["qpoints"], dtype=float)
    qpts -= np.round(qpts)                      # wrap to [-0.5, 0.5)
    freqs = np.asarray(mesh_dict["frequencies"], dtype=float)

    qnorm = np.linalg.norm(qpts, axis=1)
    near = qnorm < radius
    if not near.any():
        raise ValueError(f"no mesh q-points with |q_frac| < {radius}; "
                         f"densify mesh={tuple(mesh)}")
    candidates = []
    for qn, f in zip(qnorm[near], freqs[near]):
        if qn < 1e-8:                           # exact Gamma
            f = f[np.abs(f) > 1e-3]
        candidates.append(f)
    return float(np.min(np.concatenate(candidates)))
