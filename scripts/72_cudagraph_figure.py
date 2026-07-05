#!/usr/bin/env python
"""phosbench part 2 - render results/figures/cudagraph_speedup.png.

Repo figure conventions (mirrors scripts/40_make_plots.py):
  * Okabe-Ito palette; cuEq = blue #0072B2, e3nn = vermillion #D55E00.
  * Decision-titled: the title states the deployment decision the figure informs.
  * Watermark in a reserved bottom strip OUTSIDE the axes; 200 dpi.
  * Dependency-light: numpy + matplotlib only (no torch), so it runs on the Mac
    after rsync'ing results/raw/cudagraph/ back from the GPU box.

Reads results/raw/cudagraph/bench_{cueq,e3nn}_fp32.json (produced by
71_cudagraph_bench.py) and draws grouped bars per system size:
  eager-synced | eager-free | graph-replay, with the graph/eager-synced speedup
  annotated above each cuEq graph bar.

Usage:
    python scripts/72_cudagraph_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "results" / "raw" / "cudagraph"
FIGS = REPO / "results" / "figures"

# Okabe-Ito (identical to 40_make_plots.py).
CUEQ_BLUE = "#0072B2"
E3NN_VERM = "#D55E00"
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
             "#CC79A7", "#56B4E9", "#F0E442", "#000000"]


def _setup_style():
    plt.rcParams.update({
        "axes.prop_cycle": cycler(color=OKABE_ITO),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 9,
        "axes.titlesize": 9.5,
        "legend.fontsize": 7.5,
        "figure.constrained_layout.use": True,
    })


def _save(fig, name, src=None, dpi=200):
    FIGS.mkdir(parents=True, exist_ok=True)
    if src is not None:
        eng = fig.get_layout_engine()
        if eng is not None and hasattr(eng, "set"):
            eng.set(rect=(0.0, 0.035, 1.0, 0.965))
        fig.text(0.006, 0.008, f"source: {src}", ha="left", va="bottom",
                 fontsize=6, color="0.55")
    path = FIGS / name
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"[phosbench] wrote results/figures/{name}", flush=True)
    return path


def _load(backend):
    p = RAW / f"bench_{backend}_fp32.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _load_batched():
    p = RAW / "batched_cueq_fp32.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d if d.get("parity_pass") else None


def _rows_by_size(doc):
    out = {}
    if doc is None:
        return out
    for r in doc.get("rows", []):
        if "natoms" not in r:
            continue
        out[r["natoms"]] = r
    return out


def _med(row, arm):
    v = row.get(arm)
    return v["ms_per_step_median"] if isinstance(v, dict) else None


def main() -> int:
    _setup_style()
    cueq = _rows_by_size(_load("cueq"))
    e3nn = _rows_by_size(_load("e3nn"))
    if not cueq:
        print("SKIP: no cueq bench json", file=sys.stderr)
        return 1

    batched = _load_batched()

    sizes = sorted(cueq)
    x = np.arange(len(sizes))
    w = 0.26

    ncol = 3 if batched else 2
    fig, axs = plt.subplots(
        1, ncol, figsize=(4.9 * ncol, 4.6),
        gridspec_kw={"width_ratios": [1.15, 1.0] + ([0.75] if batched else [])})
    axc, axe = axs[0], axs[1]
    axb = axs[2] if batched else None

    # ---- left: cuEq (the launch-bound production arm) ------------------------ #
    arms = [("eager_synced", "eager (per-step synced)", "0.35"),
            ("eager_free", "eager (free-running)", "0.62"),
            ("graph_synced", "CUDA graph replay", CUEQ_BLUE)]
    for j, (arm, label, color) in enumerate(arms):
        vals = [_med(cueq[s], arm) or np.nan for s in sizes]
        bars = axc.bar(x + (j - 1) * w, vals, w * 0.92, color=color, label=label,
                       edgecolor="white", linewidth=0.4)
        axc.bar_label(bars, fmt="%.1f", fontsize=6.5, padding=1)
    # speedup annotations: label sits in whitespace above the eager bars, arrow
    # drops from the label down to the (much shorter) graph bar it explains.
    top = max(_med(cueq[s], "eager_synced") for s in sizes)
    for i, s in enumerate(sizes):
        su = cueq[s].get("speedup_graph_vs_eager_synced")
        g = _med(cueq[s], "graph_synced")
        if su and g and su >= 1.3:
            # arrow only where the graph is a decisive win; the 1.1x at 2,944 is
            # left un-arrowed (kernels dominate there - that IS the point).
            axc.annotate(f"{su:.1f}x\nfaster", xy=(x[i] + w, g + top * 0.03),
                         xytext=(x[i] + w, top * 0.70),
                         ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                         color=CUEQ_BLUE,
                         arrowprops=dict(arrowstyle="-|>", color=CUEQ_BLUE, lw=1.3))
        elif su and g:
            axc.annotate(f"{su:.2f}x\n(kernel-bound)", xy=(x[i] + w, g),
                         xytext=(x[i] + w, g + top * 0.02),
                         ha="center", va="bottom", fontsize=8, color="0.3")
    axc.set_xticks(x, [f"{s:,}" for s in sizes])
    axc.set_xlabel("phosphorene supercell (atoms)")
    axc.set_ylabel("force-eval time (ms/step)")
    axc.set_ylim(top=top * 1.20)
    axc.set_title("cuEq/fp32: a CUDA graph reclaims the host launch overhead\n"
                  "(9x at 140 atoms, fading to 1.1x once kernels dominate at 2,944)")
    axc.legend(loc="upper left")

    # ---- right: e3nn context ------------------------------------------------- #
    for j, (arm, label, color) in enumerate(
            [("eager_synced", "eager (synced)", "0.45"),
             ("graph_synced", "CUDA graph replay", E3NN_VERM)]):
        vals = [_med(e3nn.get(s, {}), arm) or np.nan for s in sizes] if e3nn else \
               [np.nan] * len(sizes)
        bars = axe.bar(x + (j - 0.5) * w, vals, w * 0.92, color=color, label=label,
                       edgecolor="white", linewidth=0.4)
        axe.bar_label(bars, fmt="%.0f", fontsize=6.5, padding=1)
    # mark the OOM where e3nn graph capture failed (the missing red bar)
    for i, s in enumerate(sizes):
        row = e3nn.get(s, {})
        if row.get("graph_synced") is None and "eager_synced" in row:
            axe.annotate("graph capture\nOOM: private\npool + e3nn\n7.4 GiB > 12 GB",
                         xy=(x[i] + 1.15 * w, _med(row, "eager_synced") * 0.42),
                         ha="center", va="center", fontsize=7, color=E3NN_VERM,
                         fontweight="bold", zorder=10,
                         bbox=dict(boxstyle="round,pad=0.35", fc="#FBE7DC",
                                   ec=E3NN_VERM, lw=1.0))
    axe.set_xticks(x, [f"{s:,}" for s in sizes])
    axe.set_xlim(x[0] - 0.6, x[-1] + 0.75)
    axe.set_xlabel("phosphorene supercell (atoms)")
    axe.set_ylabel("force-eval time (ms/step)")
    axe.set_title("e3nn/fp32 is kernel-bound, not launch-bound:\n"
                  "the graph barely helps (1.0-1.2x) and OOMs at 2,944")
    axe.legend(loc="upper left")

    # ---- right (stretch): batched arm, per-cell cost ------------------------- #
    if axb is not None:
        N = batched["config"]["n_replicas"]
        labels = ["eager\nbatched", f"{N}x single-\ncell graphs", "one batched\ngraph"]
        vals = [batched["eager_batched"]["ms_per_step_median"] / N,
                batched["ms_per_cell_single_graph"],
                batched["ms_per_cell_batched_graph"]]
        colors = ["0.45", "#56B4E9", CUEQ_BLUE]
        bx = np.arange(3)
        bars = axb.bar(bx, vals, 0.62, color=colors, edgecolor="white", linewidth=0.4)
        axb.bar_label(bars, fmt="%.2f", fontsize=7.5, padding=2)
        su = batched.get("speedup_batched_vs_Nsingles")
        axb.annotate(f"{su:.1f}x\nvs {N} singles", xy=(2, vals[2]),
                     xytext=(2, max(vals) * 0.72), ha="center", va="bottom",
                     fontsize=9, fontweight="bold", color=CUEQ_BLUE,
                     arrowprops=dict(arrowstyle="-|>", color=CUEQ_BLUE, lw=1.2))
        axb.set_xticks(bx, labels, fontsize=7.5)
        axb.set_ylabel("cost per 140-atom cell (ms)")
        axb.set_ylim(top=max(vals) * 1.25)
        axb.set_title(f"Below break-even: pack {N} small cells\n"
                      "in ONE graph to amortise the launch")

    return 0 if _save(fig, "cudagraph_speedup.png",
                      "results/raw/cudagraph/bench_*_fp32.json + batched_*.json") else 1


if __name__ == "__main__":
    raise SystemExit(main())
