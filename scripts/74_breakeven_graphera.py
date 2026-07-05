#!/usr/bin/env python3
"""Part 2 figure: what CUDA-graph capture does to the break-even chart.

Two panels from the Part 2 fixed-topology force-eval benchmarks
(results/raw/cudagraph/bench_{cueq,e3nn}_fp32*.json):
  left  — absolute ms/step vs atoms, eager vs graph, both backends;
  right — the cuEq advantage (e3nn/cuEq per-step ratio): the eager curve
          crosses 1 (finding 1's break-even, as seen by THIS harness);
          the graph-era curve never does.

Honest scope: this is the bare fixed-topology force-eval harness of Part 2 —
NOT Part 1's sweep harness (which includes ASE calculator + neighbour-list
rebuild per call), so the eager crossover position here is not expected to
coincide numerically with finding 1's 373/454-atom values. The point is the
qualitative transformation of the same decision chart.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw" / "cudagraph"
OUT = ROOT / "results" / "figures" / "breakeven_graphera.png"

# repo palette (Okabe-Ito): cuEq blue, e3nn vermillion
C_CUEQ, C_E3NN = "#0072B2", "#D55E00"
C_EAGER_RATIO, C_GRAPH_RATIO = "#666666", "#0072B2"


def load(backend: str) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for name in (f"bench_{backend}_fp32.json", f"bench_{backend}_fp32_extra.json"):
        p = RAW / name
        if not p.is_file():
            continue
        for r in json.loads(p.read_text())["rows"]:
            rows[r["natoms"]] = {
                "eager": r["eager_synced"]["ms_per_step_median"],
                "graph": (r.get("graph_synced") or {}).get("ms_per_step_median"),
            }
    return dict(sorted(rows.items()))


cueq, e3nn = load("cueq"), load("e3nn")
sizes = sorted(set(cueq) & set(e3nn))

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(11.5, 4.6), dpi=200, constrained_layout=True
)
fig.get_layout_engine().set(rect=(0, 0.04, 1, 0.96))  # bottom strip for watermark

# ---- panel 1: absolute per-step time --------------------------------------
for rows, color, label in ((e3nn, C_E3NN, "e3nn"), (cueq, C_CUEQ, "cuEq")):
    xs = [n for n in sizes]
    ax1.plot(xs, [rows[n]["eager"] for n in xs], "--o", color=color, ms=4,
             lw=1.4, alpha=0.55, label=f"{label} eager")
    gx = [n for n in xs if rows[n]["graph"]]
    ax1.plot(gx, [rows[n]["graph"] for n in gx], "-s", color=color, ms=4,
             lw=1.8, label=f"{label} + CUDA graph")
# e3nn graph-capture OOM marker at 2,944
ax1.plot([2944], [e3nn[2944]["eager"]], "x", color=C_E3NN, ms=9, mew=2.2)
ax1.annotate("graph capture\nOOM @2,944", xy=(2944, e3nn[2944]["eager"]),
             xytext=(0.97, 0.66), textcoords="axes fraction", ha="right",
             fontsize=7.5, color=C_E3NN,
             arrowprops=dict(arrowstyle="-", color=C_E3NN, lw=0.8))
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("atoms")
ax1.set_ylabel("force-eval time (ms/step)")
ax1.set_title("cuEq eager sits on a flat launch-overhead floor\n"
              "(~17 ms through 1,408 atoms) — the graph removes it")
ax1.legend(fontsize=7.5, loc="upper left")
ax1.grid(alpha=0.25, which="both")

# ---- panel 2: the break-even chart, eager era vs graph era ------------------
r_eager = [(n, e3nn[n]["eager"] / cueq[n]["eager"]) for n in sizes]
r_graph = [(n, e3nn[n]["graph"] / cueq[n]["graph"])
           for n in sizes if e3nn[n]["graph"] and cueq[n]["graph"]]
ax2.axhline(1.0, color="k", lw=0.9)
ax2.axhspan(0.0, 1.0, color=C_E3NN, alpha=0.07)
ax2.plot(*zip(*r_eager), "--o", color=C_EAGER_RATIO, ms=4, lw=1.5,
         label="eager dispatch (Part 1 regime)")
ax2.plot(*zip(*r_graph), "-s", color=C_GRAPH_RATIO, ms=5, lw=2.0,
         label="CUDA-graph capture (Part 2)")
# annotations
ax2.annotate("eager break-even:\ncuEq loses below the line",
             xy=(140, r_eager[1][1]), xytext=(75, 0.28), fontsize=7.5,
             color=C_EAGER_RATIO,
             arrowprops=dict(arrowstyle="-", color=C_EAGER_RATIO, lw=0.8))
ax2.annotate("with graph capture the crossover\nnever happens (×4.6–11.9)",
             xy=(140, dict(r_graph)[140]), xytext=(64, 2.6), fontsize=8,
             color=C_GRAPH_RATIO, fontweight="bold",
             arrowprops=dict(arrowstyle="-", color=C_GRAPH_RATIO, lw=0.8))
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("atoms")
ax2.set_ylabel("cuEq advantage (e3nn / cuEq per-step time)")
ax2.set_title("The same decision chart, before and after:\n"
              "finding 1's break-even collapses under graph capture")
ax2.legend(fontsize=7.5, loc="lower right")
ax2.grid(alpha=0.25, which="both")

fig.text(0.005, 0.005,
         "source: results/raw/cudagraph/bench_*_fp32*.json | fixed-topology "
         "force-eval harness (Part 2), not the Part 1 sweep harness",
         fontsize=6, color="0.55", ha="left", va="bottom")
fig.savefig(OUT)
print(f"wrote {OUT}")
for n, r in r_eager:
    print(f"  eager  {n:>5}: {r:5.2f}")
for n, r in r_graph:
    print(f"  graph  {n:>5}: {r:5.2f}")
