#!/usr/bin/env python
"""B-1 - parse LAMMPS LJ-melt logs into a comm/compute time-share table + plot.

Reads every results/raw/mpi/lj_np<K>.log written by 60_mpi_comm_share.sh and
extracts LAMMPS's own loop-timing breakdown:

    MPI task timing breakdown:
    Section |  min time  |  avg time  |  max time  |%varavg| %total
    ---------------------------------------------------------------
    Pair    | ...
    Neigh   | ...
    Comm    | ...
    Output  | ...
    Modify  | ...
    Other   | ...

plus the loop wall time ("Loop time of <sec> on <P> procs ...") and atom count.
The %total column is the share of loop time each section took; we keep it as-is
(that is exactly the comm/compute attribution an SA reads off a customer run).

The SA reading this demonstrates: Pair is the compute win MPI buys you; Comm is
the tax you pay for it. As ranks rise on a fixed problem (strong scaling), the
per-rank subdomain shrinks, its surface/volume ratio grows, and Comm% climbs -
eventually eating the Pair-time win. The rank count where Comm% overtakes the
marginal Pair speedup is where scale-out stops paying, and is the single number
you want before committing a customer to more nodes/GPUs.

SCOPE: classical Lennard-Jones, one node, one run per rank count. NOT MACE. The
same breakdown table is what LAMMPS prints for an ML-IAP (MACE) run via LAMMPS
ML-IAP / pair_style mace, so the methodology transfers; the numbers here are LJ.

Outputs (repo pattern: raw/ gitignored, csv+png in figures/ carry the result):
  results/figures/mpi_comm_share.csv  - parsed numbers, one row per rank count
  results/figures/mpi_comm_share.png  - stacked comm/compute share bars vs ranks
                                        with a wall-time overlay on a twin axis

Exit codes: 0 ok; 2 no parseable logs found.

Usage:
    python scripts/61_parse_lammps_log.py [--dir results/raw/mpi]
                                          [--no-plot]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIGS = REPO / "results" / "figures"

# Okabe-Ito palette (matches scripts/40_make_plots.py) - colour-vision safe.
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
             "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

# Section rows we split loop time into. "compute" = Pair+Neigh (the force work
# MPI parallelises); "comm" = Comm (the parallelisation tax); the rest is
# bookkeeping. Order here is the stack order (bottom -> top) in the figure.
SECTIONS = ["Pair", "Neigh", "Comm", "Output", "Modify", "Other"]
SECTION_COLOR = {
    "Pair":   OKABE_ITO[2],   # green  - compute (the win)
    "Neigh":  OKABE_ITO[5],   # sky    - compute (neighbour build)
    "Comm":   OKABE_ITO[3],   # orange - communication (the tax)
    "Output": OKABE_ITO[6],   # yellow - I/O
    "Modify": OKABE_ITO[4],   # purple - integrator/fixes
    "Other":  "0.7",          # grey   - unattributed
}

# "Section |  min time  |  avg time  |  max time  |%varavg| %total"
# row:  "Pair    | 5.1 | 5.2 | 5.3 | 1.2 | 74.50"
ROW_RE = re.compile(
    r"^\s*(Pair|Neigh|Comm|Output|Modify|Other)\s*\|"
    r"\s*([0-9eE.+-]+)?\s*\|"       # min (may be blank for Other)
    r"\s*([0-9eE.+-]+)\s*\|"        # avg
    r"\s*([0-9eE.+-]+)?\s*\|"       # max (may be blank)
    r"\s*([0-9eE.+-]+)?\s*\|"       # %varavg (may be blank)
    r"\s*([0-9eE.+-]+)\s*$"         # %total
)
# "Loop time of 12.34 on 4 procs for 800 steps with 500000 atoms"
LOOP_RE = re.compile(
    r"Loop time of\s+([0-9eE.+-]+)\s+on\s+(\d+)\s+procs\s+for\s+(\d+)\s+steps"
    r"\s+with\s+(\d+)\s+atoms",
    re.I,
)
# "Total wall time: 0:00:14"
WALL_RE = re.compile(r"Total wall time:\s*(\d+):(\d+):(\d+)")
NP_FROM_NAME = re.compile(r"lj_np(\d+)\.log$", re.I)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_log(path: Path) -> dict | None:
    """Extract one run's breakdown. Returns None if no timing table found."""
    text = path.read_text(errors="replace")

    loop = LOOP_RE.search(text)
    loop_time = _f(loop.group(1)) if loop else None
    procs = int(loop.group(2)) if loop else None
    steps = int(loop.group(3)) if loop else None
    atoms = int(loop.group(4)) if loop else None

    wall = WALL_RE.search(text)
    wall_s = (int(wall.group(1)) * 3600 + int(wall.group(2)) * 60
              + int(wall.group(3))) if wall else None

    pct = {}
    for line in text.splitlines():
        m = ROW_RE.match(line)
        if m:
            pct[m.group(1)] = _f(m.group(6))  # %total column
    if not pct:
        return None

    m = NP_FROM_NAME.search(path.name)
    np_ranks = procs if procs is not None else (int(m.group(1)) if m else None)

    comm = pct.get("Comm") or 0.0
    compute = (pct.get("Pair") or 0.0) + (pct.get("Neigh") or 0.0)
    return {
        "np": np_ranks,
        "atoms": atoms,
        "steps": steps,
        "loop_time_s": loop_time,
        "wall_time_s": wall_s if wall_s is not None else loop_time,
        "pct": {s: (pct.get(s) or 0.0) for s in SECTIONS},
        "comm_pct": comm,
        "compute_pct": compute,
        "src": path.name,
    }


def load_runs(mpi_dir: Path) -> list[dict]:
    runs = []
    for log in sorted(mpi_dir.glob("lj_np*.log")):
        r = parse_log(log)
        if r is None:
            print(f"[61_parse]   SKIP {log.name}: no timing table", file=sys.stderr)
            continue
        runs.append(r)
    runs.sort(key=lambda r: (r["np"] is None, r["np"] or 0))
    return runs


def write_csv(runs: list[dict], out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    # SECTIONS already emits comm_pct/pair_pct/...; the two trailing columns are
    # the derived rollups (comm_share = Comm; compute_share = Pair+Neigh) named
    # distinctly so the header has no duplicate key.
    cols = (["np", "atoms", "steps", "loop_time_s", "wall_time_s"]
            + [f"{s.lower()}_pct" for s in SECTIONS]
            + ["comm_share_pct", "compute_share_pct", "src"])
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in runs:
            w.writerow(
                [r["np"], r["atoms"], r["steps"],
                 f"{r['loop_time_s']:.4f}" if r["loop_time_s"] else "",
                 f"{r['wall_time_s']:.4f}" if r["wall_time_s"] else ""]
                + [f"{r['pct'][s]:.2f}" for s in SECTIONS]
                + [f"{r['comm_pct']:.2f}", f"{r['compute_pct']:.2f}", r["src"]]
            )
    print(f"[61_parse] wrote {out.relative_to(REPO)}")


def make_plot(runs: list[dict], out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from cycler import cycler

    plt.rcParams.update({
        "axes.prop_cycle": cycler(color=OKABE_ITO),
        "axes.grid": True, "grid.alpha": 0.3,
        "font.size": 9, "axes.titlesize": 9.5, "legend.fontsize": 7.5,
        "figure.constrained_layout.use": True,
    })

    runs = [r for r in runs if r["np"] is not None]
    x = np.arange(len(runs))
    labels = [str(r["np"]) for r in runs]

    fig, ax = plt.subplots(figsize=(7, 4.6))

    # stacked %total bars, bottom -> top in SECTIONS order
    bottom = np.zeros(len(runs))
    for s in SECTIONS:
        vals = np.array([r["pct"][s] for r in runs])
        ax.bar(x, vals, 0.6, bottom=bottom, color=SECTION_COLOR[s],
               label=s, edgecolor="white", linewidth=0.4)
        bottom += vals

    # annotate the Comm share on each bar (the number the SA is reading)
    comm_base = np.array([r["pct"]["Pair"] + r["pct"]["Neigh"] for r in runs])
    comm_vals = np.array([r["comm_pct"] for r in runs])
    for xi, base, cv in zip(x, comm_base, comm_vals):
        if cv > 0:
            ax.text(xi, base + cv / 2, f"{cv:.0f}%", ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white")

    ax.set_xticks(x, labels)
    ax.set_xlabel("MPI ranks (physical cores; SMT not used)")
    ax.set_ylabel("share of LAMMPS loop time (%)")
    ax.set_ylim(0, 100)

    # loop-time overlay on twin axis. LAMMPS's "Loop time" is the timed
    # benchmark loop at float precision - the right strong-scaling metric
    # (Total wall time has only 1-second resolution and includes setup).
    ax2 = ax.twinx()
    wall = [r["loop_time_s"] for r in runs]
    ax2.plot(x, wall, "o--", color=OKABE_ITO[7], lw=1.4, ms=5,
             label="loop time (s)")
    for xi, w in zip(x, wall):
        if w is not None:
            ax2.annotate(f"{w:.1f}s", (xi, w), textcoords="offset points",
                         xytext=(0, 7), ha="center", fontsize=7,
                         color=OKABE_ITO[7])
    ax2.set_ylabel("LAMMPS loop time (s)", color=OKABE_ITO[7])
    ax2.tick_params(axis="y", labelcolor=OKABE_ITO[7])
    ax2.grid(False)
    ymax = max((w for w in wall if w), default=1.0)
    ax2.set_ylim(0, ymax * 1.25)

    a = runs[0]["atoms"]
    st = runs[0]["steps"]
    ax.set_title("Comm share rises with rank count - the number that says when "
                 "scale-out\nstops paying (LJ melt, "
                 f"{a:,} atoms, {st} steps; classical demo)")

    # merged legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center left",
              bbox_to_anchor=(1.08, 0.5), frameon=False)

    # watermark OUTSIDE the axes (repo convention: figure-space source tag)
    fig.text(0.995, 0.005,
             "source: results/raw/mpi/lj_np*.log  |  classical LJ methodology "
             "demo (not MACE)",
             ha="right", va="bottom", fontsize=5.5, color="0.55")

    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[61_parse] wrote {out.relative_to(REPO)}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", default=str(REPO / "results/raw/mpi"))
    p.add_argument("--csv", default=str(FIGS / "mpi_comm_share.csv"))
    p.add_argument("--png", default=str(FIGS / "mpi_comm_share.png"))
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    mpi_dir = Path(args.dir)
    runs = load_runs(mpi_dir)
    if not runs:
        print(f"[61_parse] no parseable lj_np*.log under {mpi_dir} - "
              "run scripts/60_mpi_comm_share.sh first", file=sys.stderr)
        return 2

    write_csv(runs, Path(args.csv))

    # console summary (the keyNumbers)
    print(f"\n{'np':>3s} {'atoms':>9s} {'wall_s':>8s} {'Pair%':>7s} "
          f"{'Neigh%':>7s} {'Comm%':>7s} {'compute%':>9s}", flush=True)
    for r in runs:
        print(f"{r['np']!s:>3s} {r['atoms']!s:>9s} "
              f"{r['wall_time_s']:>8.2f} {r['pct']['Pair']:>7.1f} "
              f"{r['pct']['Neigh']:>7.1f} {r['comm_pct']:>7.1f} "
              f"{r['compute_pct']:>9.1f}", flush=True)

    if not args.no_plot:
        try:
            make_plot(runs, Path(args.png))
        except Exception as exc:
            print(f"[61_parse] plot skipped: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
