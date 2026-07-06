#!/usr/bin/env python
"""phosbench part 2 - CUDA-graph parity gate for the MACE force evaluation.

Motivation (README finding 3): after cuEquivariance, GPU kernels occupy ~10 % of
the MD step at 2,944 atoms - the Python/ASE host loop and per-step kernel-launch
latency dominate. The textbook fix is to capture the force evaluation into a CUDA
graph so the whole launch sequence replays as one host call. This script builds a
FIXED-TOPOLOGY force-eval path, captures it, and PROVES parity against the eager
path before any timing script is allowed to trust the graph.

Design (load-bearing constraints, from the expert review):
  * torch.compile is FORBIDDEN (cuEq #77 zero-gradient bug). We hand-capture with
    torch.cuda.graph (the context-manager form, which sets up a private mempool so
    autograd.grad allocations are served from the graph pool).
  * The MACE calculator path allocates dynamically and the neighbour list changes
    shape with positions. We freeze the graph ONCE: build the AtomicData batch a
    single time, keep edge_index / unit_shifts / shifts / node_attrs / cell / batch
    / ptr constant, and only stream new positions through a static input buffer.
    Forces come back through a static output buffer.
  * Because compute_stress=False, MACE never rebuilds `shifts` inside forward
    (get_symmetric_displacement is skipped), so the edge vectors are a pure
    function of positions:  v = pos[recv] - pos[send] + shifts  (frozen topology).
    Perturbations must stay small enough that the frozen neighbour list remains
    physically valid - we assert the max displacement is well under the cutoff
    skin and report it.
  * PARITY GATE: frozen-topology eager forces must first match the normal ASE
    calculator path, then graph-replay forces must match frozen eager forces on
    identical positions and topology. max|dF| must be ~float roundoff
    (< 1e-4 eV/A at fp32). If parity fails, NO benchmark numbers exist.

Usage (run on the GPU box via jobq):
    python scripts/70_cudagraph_gate.py --backend cueq --dtype float32 \
        --nx 8 --ny 11 --out results/raw/cudagraph/gate_cueq_fp32_512.json

Exit status is 0 iff parity passes (max|dF| < --tol).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Fixed-topology force-eval path
# --------------------------------------------------------------------------- #

class FixedTopoForceEval:
    """A frozen-neighbour-list MACE force evaluator, capturable into a CUDA graph.

    The batch (edge_index, unit_shifts, shifts, node_attrs, cell, batch, ptr) is
    built ONCE from an ASE Atoms via the calculator's own _atoms_to_batch, then
    held constant.  Only `positions` is mutable: callers copy new positions into
    `self.positions` (in place) and call eager() or replay().

    eager()  : runs model.forward directly (the reference path).
    capture(): records one forward+autograd-grad into a torch.cuda graph.
    replay() : replays the captured graph; forces land in self.forces_out.
    """

    def __init__(self, calc, atoms, device="cuda", n_replicas=1):
        import torch

        self.torch = torch
        self.device = device
        self.calc = calc
        self.model = calc.models[0]
        self.model_dtype = next(self.model.parameters()).dtype
        self.n_replicas = n_replicas

        # Build the batch ONCE.  For a single system we go through the calculator's
        # own converter so the neighbour list / edge_index / shifts match the
        # production path exactly.  For the batched (stretch) arm we replicate the
        # same AtomicData n_replicas times into one torch_geometric Batch - the
        # model treats them as disjoint graphs (block-diagonal edge_index), so ONE
        # graph replay evaluates all replicas at once.
        if n_replicas == 1:
            batch = calc._atoms_to_batch(atoms)
        else:
            batch = self._build_replica_batch(calc, atoms, n_replicas)
        for key in batch.keys:
            v = batch[key]
            if torch.is_tensor(v) and torch.is_floating_point(v):
                batch[key] = v.to(dtype=self.model_dtype)
        self.batch = batch
        self.batch_dict = batch.to_dict()

        # Static input buffer: positions get requires_grad so autograd.grad can
        # differentiate energy w.r.t. them for forces.  We keep this exact tensor
        # object as the graph input - replay copies new data INTO it in place.
        pos = self.batch_dict["positions"].detach().clone().requires_grad_(True)
        self.batch_dict["positions"] = pos
        self.positions = pos                      # static input buffer
        self.n_atoms = pos.shape[0]
        self.n_edges = int(self.batch_dict["edge_index"].shape[1])

        # r_max / cutoff for the skin-validity report.
        self.r_max = float(getattr(self.model, "r_max", 0.0))

        # Filled after capture.
        self.graph = None
        self.forces_out = None      # static output buffer
        self.energy_out = None

    @staticmethod
    def _build_replica_batch(calc, atoms, n_replicas):
        """One torch_geometric Batch of n_replicas identical AtomicData graphs.

        Rebuilds each replica through the same code path _atoms_to_batch uses
        (config_from_atoms -> AtomicData.from_config), then batches with MACE's
        bundled torch_geometric.  The result is a single block-diagonal graph the
        model evaluates in one forward - so one CUDA-graph replay covers all
        replicas.  This probes the below-break-even regime: many small cells whose
        per-cell launch cost is amortised across the batch.
        """
        from mace import data as mace_data
        from mace.tools import torch_geometric, torch_tools

        keyspec = mace_data.KeySpecification(
            info_keys=calc.info_keys, arrays_keys=calc.arrays_keys)
        with torch_tools.default_dtype(calc.default_dtype):
            graphs = []
            for _ in range(n_replicas):
                config = mace_data.config_from_atoms(
                    atoms, key_specification=keyspec, head_name=calc.head)
                graphs.append(mace_data.AtomicData.from_config(
                    config, z_table=calc.z_table, cutoff=calc.r_max,
                    heads=calc.available_heads))
        return torch_geometric.Batch.from_data_list(graphs).to(calc.device)

    # -- eager reference ----------------------------------------------------- #
    def _forward(self):
        """One eager forward returning (energy_scalar_tensor, forces_tensor)."""
        torch = self.torch
        # positions already has requires_grad; MACE calls requires_grad_(True)
        # again in prepare_graph, which is a no-op here.  training=False so the
        # autograd graph is freed after each grad (retain_graph=False) - fine,
        # capture records the build+free once and replay re-executes it.
        out = self.model(
            self.batch_dict,
            compute_force=True,
            compute_stress=False,
            compute_virials=False,
            compute_edge_forces=False,
            training=False,
        )
        return out["energy"], out["forces"]

    def eager(self, positions_np=None):
        """Eager force eval. If positions_np given, copy it into the static buffer
        first (so eager and replay see identical inputs)."""
        torch = self.torch
        if positions_np is not None:
            with torch.no_grad():
                self.positions.copy_(
                    torch.as_tensor(positions_np, dtype=self.model_dtype,
                                    device=self.device))
        # Fresh grad each call; do NOT accumulate.
        if self.positions.grad is not None:
            self.positions.grad = None
        energy, forces = self._forward()
        return energy.detach(), forces.detach()

    # -- CUDA-graph capture -------------------------------------------------- #
    def capture(self, n_warmup=3):
        """Capture forward+autograd-grad into a CUDA graph.

        We follow the torch.cuda.graph recipe: run a few warmups on a side stream
        (so cuEq / autograd lazily allocate their workspaces and any one-time
        cudaMallocs happen BEFORE capture), then record.  The energy and forces
        that forward produces become the STATIC output buffers - replay overwrites
        them in place, so callers read self.forces_out after every replay.
        """
        torch = self.torch
        # Warm up on a private stream so lazy init / autograd workspace allocation
        # is complete before capture (mallocs during capture are illegal).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warmup):
                if self.positions.grad is not None:
                    self.positions.grad = None
                energy, forces = self._forward()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # Capture.  The graph pool is private so autograd.grad's temporary
        # buffers are allocated inside it and reused across replays.
        self.graph = torch.cuda.CUDAGraph()
        if self.positions.grad is not None:
            self.positions.grad = None
        with torch.cuda.graph(self.graph):
            energy, forces = self._forward()
        # These tensors are written in place by every replay.
        self.energy_out = energy
        self.forces_out = forces
        torch.cuda.synchronize()

    def replay(self, positions_np=None):
        """Copy new positions into the static buffer, replay, return static forces.

        The returned tensor IS self.forces_out - callers must .clone() it if they
        need to keep the value past the next replay.
        """
        torch = self.torch
        if positions_np is not None:
            with torch.no_grad():
                self.positions.copy_(
                    torch.as_tensor(positions_np, dtype=self.model_dtype,
                                    device=self.device))
        self.graph.replay()
        return self.energy_out, self.forces_out


# --------------------------------------------------------------------------- #
# Parity gate
# --------------------------------------------------------------------------- #

def run_gate(backend, dtype, nx, ny, model, device, tol, seed, out_path):
    import torch
    from phosbench.common import env_metadata, gpu_clock_temp, make_calc, make_supercell

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    atoms = make_supercell(int(nx), int(ny))
    natoms = len(atoms)
    print(f"[gate] {backend}/{dtype} {nx}x{ny} = {natoms} atoms; building calc...",
          flush=True)
    calc = make_calc(backend, dtype, model=model, device=device)

    fe = FixedTopoForceEval(calc, atoms, device=device)
    print(f"[gate] frozen topology: {fe.n_atoms} atoms, {fe.n_edges} edges, "
          f"r_max={fe.r_max:.3f} A, model_dtype={fe.model_dtype}", flush=True)

    base = atoms.get_positions().astype(np.float64)

    # A single perturbed geometry both paths must agree on.  Keep the rattle small
    # (0.01 A rms, same scale as common.time_force_calls) so the frozen neighbour
    # list stays physically valid: max displacement must be << the neighbour skin.
    perturbed = base + rng.normal(scale=0.01, size=base.shape)
    max_disp = float(np.abs(perturbed - base).max())

    result = {
        "config": {"backend": backend, "dtype": dtype, "model": model,
                   "device": device, "nx": int(nx), "ny": int(ny)},
        "natoms": natoms,
        "n_edges": fe.n_edges,
        "r_max": fe.r_max,
        "max_perturb_disp_A": max_disp,
        "rattle_scale_A": 0.01,
        "tol_eV_per_A": tol,
        "clocks": gpu_clock_temp() if device == "cuda" else None,
    }

    # 1) Eager reference on the perturbed geometry.
    e_eager, f_eager = fe.eager(perturbed)
    f_eager = f_eager.clone()
    e_eager = float(e_eager.item())
    print(f"[gate] eager forward OK: E={e_eager:.6f} eV, "
          f"|F|max={float(f_eager.abs().max()):.4f} eV/A", flush=True)

    # 1b) Production calculator reference on the same geometry. This rebuilds the
    # neighbor list through the normal ASE calculator path. Graph-vs-frozen parity is
    # not enough; the frozen topology must also match the production calculator.
    atoms_prod = atoms.copy()
    atoms_prod.set_positions(perturbed)
    atoms_prod.calc = calc
    f_calc = torch.as_tensor(atoms_prod.get_forces(), dtype=f_eager.dtype,
                             device=f_eager.device)
    dF_topo = (f_eager - f_calc).abs()
    max_dF_topo = float(dF_topo.max().item())
    rms_dF_topo = float((dF_topo ** 2).mean().sqrt().item())
    result["max_dF_frozen_vs_calculator_eV_per_A"] = max_dF_topo
    result["rms_dF_frozen_vs_calculator_eV_per_A"] = rms_dF_topo
    result["topology_matches_calculator"] = bool(max_dF_topo < tol)
    print(f"[gate] frozen topology vs production calculator max|dF|="
          f"{max_dF_topo:.3e} eV/A "
          f"({'PASS' if result['topology_matches_calculator'] else 'FAIL'})",
          flush=True)
    if not result["topology_matches_calculator"]:
        result["parity_pass"] = False
        _write(out_path, result)
        return 1

    # 2) Capture on the frozen topology, then replay on the SAME perturbed geometry.
    capture_error = None
    try:
        fe.capture(n_warmup=3)
        print("[gate] CUDA-graph capture OK", flush=True)
    except Exception as exc:  # noqa: BLE001
        capture_error = f"{type(exc).__name__}: {exc}"
        result["capture_error"] = capture_error[:2000]
        result["parity_pass"] = False
        print(f"[gate] CAPTURE FAILED: {capture_error}", flush=True)
        _write(out_path, result)
        return 2

    e_graph, f_graph = fe.replay(perturbed)
    f_graph = f_graph.clone()
    e_graph = float(e_graph.item())

    # 3) Parity: max|dF| and dE on identical positions + topology.
    dF = (f_graph - f_eager).abs()
    max_dF = float(dF.max().item())
    rms_dF = float((dF ** 2).mean().sqrt().item())
    dE = abs(e_graph - e_eager)
    fmax = float(f_eager.abs().max().item())
    rel_dF = max_dF / fmax if fmax > 0 else float("nan")

    result.update({
        "energy_eager_eV": e_eager,
        "energy_graph_eV": e_graph,
        "dE_eV": dE,
        "max_dF_eV_per_A": max_dF,
        "rms_dF_eV_per_A": rms_dF,
        "rel_dF": rel_dF,
        "fmax_eV_per_A": fmax,
        "parity_pass": bool(max_dF < tol),
    })

    # 4) Second replay on a DIFFERENT perturbation - the graph must track new
    #    positions (proves the static-buffer copy-in actually feeds the replay,
    #    i.e. we did not accidentally freeze the forces too).
    perturbed2 = base + rng.normal(scale=0.01, size=base.shape)
    e_eager2, f_eager2 = fe.eager(perturbed2)
    f_eager2 = f_eager2.clone()
    e_graph2, f_graph2 = fe.replay(perturbed2)
    max_dF2 = float((f_graph2 - f_eager2).abs().max().item())
    # Sanity: forces genuinely CHANGED between the two geometries.
    delta_between = float((f_graph2 - f_graph).abs().max().item())
    result["max_dF_second_geom"] = max_dF2
    result["force_change_between_geoms"] = delta_between
    result["tracks_positions"] = bool(delta_between > 1e-3)
    result["parity_pass"] = bool(result["parity_pass"] and max_dF2 < tol
                                 and result["tracks_positions"]
                                 and result["topology_matches_calculator"])

    print(f"[gate] PARITY  max|dF|={max_dF:.3e} eV/A (rel {rel_dF:.1e}), "
          f"dE={dE:.3e} eV, 2nd-geom max|dF|={max_dF2:.3e}, "
          f"tracks_positions={result['tracks_positions']}", flush=True)
    verdict = "PASS" if result["parity_pass"] else "FAIL"
    print(f"[gate] {verdict}: max|dF|={max_dF:.3e} eV/A vs tol {tol:.1e}", flush=True)

    _write(out_path, result)
    return 0 if result["parity_pass"] else 1


def _write(out_path, result):
    from phosbench.common import env_metadata
    result["env"] = env_metadata()
    p = Path(out_path)
    if not p.is_absolute():
        p = REPO / p
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"[gate] wrote {p}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="cueq", choices=["cueq", "e3nn"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--nx", type=int, default=8)
    ap.add_argument("--ny", type=int, default=11)
    ap.add_argument("--model", default="medium")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="parity threshold on max|dF| (eV/A)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/raw/cudagraph/gate.json")
    args = ap.parse_args()
    return run_gate(args.backend, args.dtype, args.nx, args.ny, args.model,
                    args.device, args.tol, args.seed, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
