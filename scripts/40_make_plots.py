#!/usr/bin/env python
"""Stage E - render every figure from whatever exists under results/raw/.

Design rules:

  * Decision-titled: every figure title states the deployment decision it
    informs, not the quantity it shows (PROTOCOL.md, Stage E).
  * Tolerant: each figure is independent and wrapped in try/except - a missing
    input prints "SKIP <figure>: missing <file>" and the rest still render.
    Figures are rebuilt from scratch on every invocation, so this script can
    run after any subset of stages A-D has completed.
  * Dependency-light: numpy + matplotlib only (Agg). No torch import, so the
    analysis can run on a laptop with results/ rsync'ed off the GPU box.

Outputs: results/figures/*.png at 150 dpi plus oom_boundary.csv.

Usage:
    python scripts/40_make_plots.py [--only fig1,fig5,...] [--smoke]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # headless box; must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "results" / "raw"
FIGS = REPO / "results" / "figures"

THZ_TO_CM1 = 33.356
VRAM_LIMIT_MIB = 12288.0            # RTX 3080 Ti physical VRAM
LIT_C11_NPM, LIT_C22_NPM = 24.0, 103.0  # DFT lit., monolayer (armchair/zigzag)

# Okabe-Ito palette: distinguishable under the common color-vision deficiencies.
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
             "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

# Stable colors per cell so a config looks the same in every figure.
CELL_COLORS = {
    "e3nn/float64": OKABE_ITO[0],   # reference
    "e3nn/float32": OKABE_ITO[1],   # precision arm
    "cueq/float32": OKABE_ITO[2],   # production candidate
    "cueq/float64": OKABE_ITO[3],   # broken upstream - tolerated, never required
}
MODEL_MARKERS = {"medium": "o", "medium-mpa-0": "s", "medium-omat-0": "^",
                 "small": "v", "large": "D"}


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


def _cell_color(cell: str) -> str:
    return CELL_COLORS.get(cell, OKABE_ITO[4 + sum(map(ord, cell)) % 4])


def _model_marker(model: str) -> str:
    return MODEL_MARKERS.get(model, "o")


# --------------------------------------------------------------------------- #
# Tolerant loaders
# --------------------------------------------------------------------------- #

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(_rel(path))
    return path


def _first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(_rel(paths[0]))


def _first(d: dict, *keys):
    """First non-None value among candidate key spellings (schema tolerance)."""
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None


def _load_sweep() -> list[dict]:
    """Merge all sweep*.jsonl files, last record winning per measurement.

    The sweep appends on rerun, so duplicates are expected; keeping the last
    occurrence means a fixed rerun supersedes an earlier crash.
    """
    files = sorted(RAW.glob("sweep*.jsonl"))
    if not files:
        raise FileNotFoundError("results/raw/sweep*.jsonl")
    recs: dict[tuple, dict] = {}
    for f in files:
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # half-written tail of a killed run
            c = rec.get("config", {})
            key = tuple(c.get(k) for k in
                        ("backend", "dtype", "model", "device", "mode", "nx", "ny"))
            recs[key] = rec
    return list(recs.values())


def _groups(recs: list[dict],
            keys=("backend", "dtype", "model", "device")) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = {}
    for r in recs:
        c = r.get("config", {})
        out.setdefault(tuple(c.get(k) for k in keys), []).append(r)
    for rows in out.values():
        rows.sort(key=lambda r: r.get("natoms", 0))
    return out


def _parse_cell_tag(name: str) -> str | None:
    """'backend/dtype' from a filename like band_cueq_fp32_d0.010.npz."""
    name = name.lower()
    backend = "cueq" if "cueq" in name else ("e3nn" if "e3nn" in name else None)
    dtype = ("float64" if ("float64" in name or "fp64" in name) else
             "float32" if ("float32" in name or "fp32" in name) else None)
    return f"{backend}/{dtype}" if backend and dtype else None


def _parse_disp(name: str) -> float | None:
    m = re.search(r"(?:^|[_-])d(?:isp)?[_-]?([0-9]+(?:\.[0-9]+)?)", name.lower())
    return float(m.group(1)) if m else None


def _npz_text(z, *keys) -> str | None:
    for k in keys:
        if k in z.files:
            v = np.asarray(z[k]).ravel()
            if v.size:
                return str(v[0])
    return None


def _npz_scalar(z, *keys) -> float | None:
    for k in keys:
        if k in z.files:
            try:
                return float(np.asarray(z[k]).ravel()[0])
            except (TypeError, ValueError, IndexError):
                pass
    return None


def _npz_pick(z, *keys) -> np.ndarray:
    for k in keys:
        if k in z.files:
            return np.asarray(z[k], dtype=float)
    raise KeyError(f"none of {keys} in npz (has {z.files})")


def _save(fig, name: str, src=None) -> Path:
    FIGS.mkdir(parents=True, exist_ok=True)
    if src is not None:
        fig.text(0.995, 0.005, f"source: {src}", ha="right", va="bottom",
                 fontsize=5.5, color="0.55")
    path = FIGS / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[phosbench] wrote {_rel(path)}", flush=True)
    return path


# --------------------------------------------------------------------------- #
# fig1 - throughput vs size
# --------------------------------------------------------------------------- #

def fig_throughput() -> list[Path]:
    recs = [r for r in _load_sweep()
            if r.get("config", {}).get("mode") == "md"]
    if not recs:
        raise FileNotFoundError("md-mode records in results/raw/sweep*.jsonl")

    fig, ax = plt.subplots(figsize=(7, 5))
    for (backend, dtype, model, device), rows in sorted(_groups(recs).items()):
        ok = [r for r in rows if not r.get("error") and r.get("ns_per_day_1fs")]
        if not ok:
            continue
        cell = f"{backend}/{dtype}"
        x = [r["natoms"] for r in ok]
        y = [r["ns_per_day_1fs"] for r in ok]
        label = f"{cell} {model}" + ("" if device == "cuda" else f" [{device}]")
        ax.plot(x, y, lw=1.3, ms=4, color=_cell_color(cell),
                marker=_model_marker(model),
                linestyle="-" if device == "cuda" else ":", label=label)
        if any(r.get("error") == "oom" for r in rows):
            # x at the last size that still fit - the practical ceiling
            ax.plot([x[-1]], [y[-1]], "x", ms=11, mew=2.2,
                    color=_cell_color(cell))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("atoms")
    ax.set_ylabel("MD throughput (ns/day @ 1 fs)")
    ax.set_title("Pick backend+precision by system size; x marks the last size "
                 "that fits in 12 GiB")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="x", ls="none", mew=2, color="k"))
    labels.append("last size before OOM")
    ax.legend(handles, labels, ncol=2)  # ncol: works on matplotlib < 3.6 too
    return [_save(fig, "throughput_vs_size.png", "sweep*.jsonl (md mode)")]


# --------------------------------------------------------------------------- #
# fig2 - cueq/e3nn speedup and break-even
# --------------------------------------------------------------------------- #

def _crossings(x, y, level=1.0) -> list[float]:
    """x values where the piecewise-linear y(log x) crosses `level`."""
    out = []
    for i in range(len(y) - 1):
        y0, y1 = y[i] - level, y[i + 1] - level
        if y0 == 0.0:
            out.append(float(x[i]))
        elif y0 * y1 < 0:
            f = y0 / (y0 - y1)
            out.append(float(np.exp(np.log(x[i])
                                    + f * (np.log(x[i + 1]) - np.log(x[i])))))
    return out


def fig_breakeven() -> list[Path]:
    series: dict[tuple, dict[str, dict[int, float]]] = {}
    for r in _load_sweep():
        c = r.get("config", {})
        cell = f"{c.get('backend')}/{c.get('dtype')}"
        if (r.get("error") or not r.get("ms_per_step_median")
                or cell not in ("e3nn/float32", "cueq/float32")):
            continue
        key = (c.get("model"), c.get("device"), c.get("mode"))
        series.setdefault(key, {}).setdefault(cell, {})[r["natoms"]] = \
            r["ms_per_step_median"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    models = sorted({k[0] for k in series})
    colors = {m: OKABE_ITO[i % len(OKABE_ITO)] for i, m in enumerate(models)}
    notes, drawn = [], 0
    for (model, device, mode), cells in sorted(series.items()):
        e3, cu = cells.get("e3nn/float32"), cells.get("cueq/float32")
        if not e3 or not cu:
            continue
        common = sorted(set(e3) & set(cu))
        if not common:
            continue
        ratio = [e3[n] / cu[n] for n in common]
        label = f"{model} {mode}" + ("" if device == "cuda" else f" [{device}]")
        ax.plot(common, ratio, lw=1.4, ms=4, marker="o",
                color=colors[model],
                linestyle="-" if mode == "force_call" else "--", label=label)
        drawn += 1
        cross = _crossings(common, ratio)
        if cross:
            ax.axvline(cross[0], color=colors[model], ls=":", lw=0.9, alpha=0.7)
            notes.append(f"{label}: break-even ~{cross[0]:,.0f} atoms")
        else:
            who = "cueq" if ratio[-1] > 1.0 else "e3nn"
            notes.append(f"{label}: no crossing ({who} faster throughout)")
    if not drawn:
        raise FileNotFoundError("matched e3nn/float32 + cueq/float32 records "
                                "in results/raw/sweep*.jsonl")

    ax.axhline(1.0, color="0.4", ls="-", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("atoms")
    ax.set_ylabel("speedup  e3nn/fp32 / cueq/fp32  (median ms/step ratio)")
    ax.set_title("Switch to cuEq only above break-even -\n"
                 "kernel crossover (solid) lands earlier than wall-clock (dashed)")
    ax.text(0.02, 0.98, "\n".join(notes), transform=ax.transAxes, va="top",
            fontsize=7, bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.9))
    ax.legend(loc="lower right")
    return [_save(fig, "speedup_breakeven.png", "sweep*.jsonl")]


# --------------------------------------------------------------------------- #
# fig3 - VRAM vs size + OOM boundary table
# --------------------------------------------------------------------------- #

def fig_vram() -> list[Path]:
    recs = _load_sweep()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = 0
    for (backend, dtype, model, device), rows in sorted(_groups(recs).items()):
        modes = sorted({r.get("config", {}).get("mode") for r in rows
                        if r.get("config", {}).get("mode")})
        mode = "md" if "md" in modes else (modes[0] if modes else None)
        sel = [r for r in rows if r.get("config", {}).get("mode") == mode]
        ok = [r for r in sel if not r.get("error") and r.get("peak_vram_mib")]
        if not ok:
            continue
        cell = f"{backend}/{dtype}"
        x = [r["natoms"] for r in ok]
        y = [r["peak_vram_mib"] for r in ok]
        ax.plot(x, y, lw=1.3, ms=4, color=_cell_color(cell),
                marker=_model_marker(model), label=f"{cell} {model} ({mode})")
        if any(r.get("error") == "oom" for r in sel):
            ax.plot([x[-1]], [y[-1]], "x", ms=11, mew=2.2,
                    color=_cell_color(cell))
        plotted += 1
    if not plotted:
        raise FileNotFoundError("records with peak_vram_mib in "
                                "results/raw/sweep*.jsonl")

    ax.axhline(VRAM_LIMIT_MIB, color="#D55E00", ls="--", lw=1.4)
    ax.text(0.01, VRAM_LIMIT_MIB, " 12 GiB (RTX 3080 Ti)", va="bottom",
            fontsize=7.5, color="#D55E00", transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_xlabel("atoms")
    ax.set_ylabel("peak VRAM (MiB, torch allocator)")
    ax.set_title("Size your production cell below the OOM boundary - cuEq's "
                 "memory saving extends it")
    ax.legend()
    out = [_save(fig, "vram_oom.png", "sweep*.jsonl")]

    # OOM boundary: one row per full config including mode, printed + CSV
    hdr = ["backend", "dtype", "model", "device", "mode",
           "last_ok_natoms", "last_ok_peak_vram_mib", "first_oom_natoms"]
    table = []
    for key, rows in sorted(_groups(recs, keys=("backend", "dtype", "model",
                                                "device", "mode")).items()):
        ok = [r for r in rows if not r.get("error")]
        oom = [r for r in rows if r.get("error") == "oom"]
        last = ok[-1] if ok else None
        peak = last.get("peak_vram_mib") if last else None
        table.append(list(key) + [
            last["natoms"] if last else "",
            f"{peak:.0f}" if peak else "",
            oom[0]["natoms"] if oom else "",
        ])
    csv_path = FIGS / "oom_boundary.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(table)
    print(f"[phosbench] wrote {_rel(csv_path)}")
    widths = [max(len(str(x)) for x in [h] + [row[i] for row in table])
              for i, h in enumerate(hdr)]
    print("OOM boundary:")
    for row in [hdr] + table:
        print("  " + "  ".join(str(x).ljust(w) for x, w in zip(row, widths)))
    out.append(csv_path)
    return out


# --------------------------------------------------------------------------- #
# fig4 - Stage A parity gates
# --------------------------------------------------------------------------- #

def fig_parity() -> list[Path]:
    path = _require(RAW / "stage_a_consistency.json")
    data = json.loads(path.read_text())
    gate = data.get("gates", {}).get("cueq_vs_e3nn_fp32")
    prec = data.get("reports", {}).get("precision_e3nn")
    pairs = [(name, d) for name, d in
             [("cueq vs e3nn\n@ fp32 (GATE)", gate),
              ("fp32 vs fp64\n(e3nn)", prec)] if d]
    if not pairs:
        raise FileNotFoundError("gates/reports in stage_a_consistency.json")

    floor = 1e-9  # log-scale guard for bitwise-identical results
    de = [max(float(d["dE_meV_per_atom"]), floor) for _, d in pairs]
    df = [max(float(d["max_dF_meV_per_A"]), floor) for _, d in pairs]
    x = np.arange(len(pairs))

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    b1 = ax.bar(x - 0.18, de, 0.34, color=OKABE_ITO[0], label="|dE| (meV/atom)")
    b2 = ax.bar(x + 0.18, df, 0.34, color=OKABE_ITO[1], label="max |dF| (meV/A)")
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.1e", fontsize=7, padding=2)
    ax.axhline(1.0, color="#D55E00", ls="--", lw=1.4,
               label="gate: 1 meV/atom, 1 meV/A")
    ax.set_yscale("log")
    ax.set_ylim(min(de + df) / 30, max(de + df + [1.0]) * 30)
    ax.set_xticks(x, [n for n, _ in pairs])
    ax.set_ylabel("deviation (log)")
    verdict = ""
    if gate is not None and "pass" in gate:
        verdict = " [GATE " + ("PASS" if gate["pass"] else "FAIL") + "]"
    ax.set_title("Backend swap is numerically free; precision is the real "
                 f"knob{verdict}")
    ax.legend(loc="upper left")
    return [_save(fig, "parity_gates.png", "stage_a_consistency.json")]


# --------------------------------------------------------------------------- #
# fig5 - phonon dispersion overlay (+ ZA zoom)
# --------------------------------------------------------------------------- #

def _load_bands() -> list[dict]:
    pdir = RAW / "phonons"
    files = sorted(pdir.glob("*.npz"))
    if not files:
        raise FileNotFoundError("results/raw/phonons/*.npz")
    bands = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        if not {"distances", "frequencies_THz", "labels",
                "label_distances"}.issubset(z.files):
            continue
        backend, dtype = _npz_text(z, "backend"), _npz_text(z, "dtype")
        cell = (f"{backend}/{dtype}" if backend and dtype
                else _npz_text(z, "cell") or _parse_cell_tag(f.name))
        bands.append({
            "cell": cell,
            "disp": _npz_scalar(z, "displacement", "displacement_A", "disp_A")
                    or _parse_disp(f.name),
            "distances": np.asarray(z["distances"], float),
            "freq_cm1": np.asarray(z["frequencies_THz"], float) * THZ_TO_CM1,
            "labels": [str(x) for x in np.asarray(z["labels"]).ravel()],
            "label_distances": np.asarray(z["label_distances"], float),
            "file": f.name,
        })
    if not bands:
        raise FileNotFoundError("band npz with distances/frequencies_THz/"
                                "labels/label_distances in results/raw/phonons/")
    return bands


def fig_phonon_dispersion() -> list[Path]:
    bands = _load_bands()
    # one band per cell, at (or nearest to) the canonical 0.01 A displacement
    by_cell: dict[str, dict] = {}
    for b in bands:
        key = b["cell"] or b["file"]
        d = 0.01 if b["disp"] is None else b["disp"]
        if key not in by_cell or abs(d - 0.01) < abs(
                (by_cell[key]["disp"] or 0.01) - 0.01):
            by_cell[key] = b
    order = sorted(by_cell, key=lambda c: (c != "e3nn/float64", c))

    fig, (ax, axz) = plt.subplots(
        1, 2, figsize=(8.5, 4.4), gridspec_kw={"width_ratios": [3, 1.2]})
    for cell in order:
        b = by_cell[cell]
        ref = (cell == "e3nn/float64")
        style = dict(color=_cell_color(cell), lw=1.3 if ref else 1.0,
                     linestyle="-" if ref else "--")
        for ax_i, cols in ((ax, range(b["freq_cm1"].shape[1])), (axz, range(3))):
            for j, col in enumerate(cols):
                ax_i.plot(b["distances"], b["freq_cm1"][:, col],
                          label=(f"{cell} (d={b['disp'] or 0.01:g} A)"
                                 if j == 0 and ax_i is ax else None), **style)

    ref_b = by_cell[order[0]]
    ticks = ref_b["label_distances"]
    names = ["$\\Gamma$" if l.upper() in ("G", "GAMMA") else l
             for l in ref_b["labels"]]
    ax.set_xticks(ticks, names)
    for t in ticks[1:-1]:
        ax.axvline(t, color="0.8", lw=0.7)
    ax.axhline(0.0, color="0.6", lw=0.7)
    ax.set_xlim(ticks[0], ticks[-1])
    ax.set_ylabel("frequency (cm$^{-1}$)")
    ax.set_title("If dashed overlays solid, cueq/fp32 spectra are trustworthy")
    ax.legend(loc="upper right")

    # ZA zoom: the soft flexural branch near Gamma is where fp32 noise bites
    gidx = next((i for i, l in enumerate(ref_b["labels"])
                 if l.upper() in ("G", "GAMMA")), len(ticks) // 2)
    d_g = ticks[gidx]
    gaps = [abs(d_g - ticks[i]) for i in (gidx - 1, gidx + 1)
            if 0 <= i < len(ticks)]
    w = 0.35 * min(gaps)
    axz.set_xlim(d_g - w, d_g + w)
    mask = np.abs(ref_b["distances"] - d_g) <= w
    lo = min(0.0, float(ref_b["freq_cm1"][mask, :3].min()))
    # scale to the soft ZA branch (col 0); TA/LA just enter from the sides
    hi = max(3.0 * float(ref_b["freq_cm1"][mask, 0].max()), 15.0)
    axz.set_ylim(lo - 3, hi)
    axz.axhline(0.0, color="0.6", lw=0.7)
    axz.set_xticks([d_g], ["$\\Gamma$"])
    axz.set_title("ZA zoom near $\\Gamma$", fontsize=8.5)
    return [_save(fig, "phonon_dispersion.png", "phonons/*.npz")]


# --------------------------------------------------------------------------- #
# fig6 - finite-displacement noise
# --------------------------------------------------------------------------- #

def _phonon_noise_rows(summary: dict) -> list[tuple[str, float, np.ndarray]]:
    """(cell, displacement_A, per-branch RMSE in cm^-1) rows, schema-tolerant."""
    rows: list[tuple[str, float, np.ndarray]] = []
    # canonical 20_phonons.py schema: rmse_vs_reference: {"cell@d0.01": {...}}
    node = summary.get("rmse_vs_reference")
    if isinstance(node, dict):
        for key, val in node.items():
            if not isinstance(val, dict):
                continue
            cell = val.get("cell") or key.split("@")[0]
            disp = _first(val, "displacement", "displacement_A", "disp_A")
            if disp is None:
                m = re.search(r"@d([0-9.]+)$", key)
                disp = float(m.group(1)) if m else None
            rmse = _first(val, "per_branch_rmse_cm1", "rmse_cm1_per_branch",
                          "overall_rmse_cm1")
            if not cell or disp is None or rmse is None:
                continue
            rows.append((cell, float(disp),
                         np.atleast_1d(np.asarray(rmse, float))))
        if rows:
            return rows
    recs = summary.get("records")
    if isinstance(recs, list):
        for r in recs:
            cell = r.get("cell") or (
                f"{r['backend']}/{r['dtype']}"
                if "backend" in r and "dtype" in r else None)
            disp = _first(r, "displacement", "displacement_A", "disp_A", "disp")
            rmse, scale = _first(r, "rmse_cm1_per_branch", "per_branch_rmse_cm1",
                                 "branch_rmse_cm1", "rmse_cm1",
                                 "band_rmse_cm1"), 1.0
            if rmse is None:
                rmse, scale = _first(r, "rmse_THz_per_branch",
                                     "rmse_THz"), THZ_TO_CM1
            if cell is None or disp is None or rmse is None:
                continue
            rows.append((cell, float(disp),
                         np.atleast_1d(np.asarray(rmse, float)) * scale))
        if rows:
            return rows
    for key in ("rmse_cm1", "cells"):  # nested {cell: {disp: rmse}} variants
        node = summary.get(key)
        if not isinstance(node, dict):
            continue
        for cell, per_disp in node.items():
            if not isinstance(per_disp, dict):
                continue
            for disp, val in per_disp.items():
                try:
                    d = float(disp)
                except (TypeError, ValueError):
                    continue
                if isinstance(val, dict):
                    val = _first(val, "rmse_cm1_per_branch", "rmse_cm1",
                                 "per_branch", "rmse")
                if val is None:
                    continue
                rows.append((cell, d, np.atleast_1d(np.asarray(val, float))))
        if rows:
            return rows
    return rows


def fig_phonon_noise() -> list[Path]:
    path = _first_existing(RAW / "phonons_summary.json",
                           RAW / "phonons" / "phonons_summary.json",
                           RAW / "phonons" / "summary.json")
    rows = [r for r in _phonon_noise_rows(json.loads(path.read_text()))
            if float(np.mean(r[2])) > 0.0]  # drop reference-vs-itself zeros
    if not rows:
        raise FileNotFoundError(f"RMSE records in {_rel(path)}")

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    floor = 1e-3
    for cell in sorted({c for c, _, _ in rows}):
        pts = sorted((d, r) for c, d, r in rows if c == cell)
        disps = [d for d, _ in pts]
        ax.plot(disps, [max(float(np.mean(r)), floor) for _, r in pts],
                marker="o", ms=5, lw=1.4, color=_cell_color(cell), label=cell)
        for d, r in pts:  # per-branch scatter shows the spread, not just mean
            ax.plot([d] * len(r), np.clip(r, floor, None), ".", ms=3,
                    alpha=0.35, color=_cell_color(cell))

    ax.set_yscale("log")
    ax.set_xlabel("finite-difference displacement (A)")
    ax.set_ylabel("per-branch RMSE vs e3nn/fp64 reference (cm$^{-1}$)")
    ax.set_xticks(sorted({d for _, d, _ in rows}))
    ax.set_title("fp32 force noise pollutes small displacements - keep "
                 "phonon workflows on fp64")
    ax.text(0.98, 0.02,
            "hybrid policy: displaced-force properties on e3nn/fp64,\n"
            "production MD on cueq/fp32",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            bbox=dict(boxstyle="round", fc="w", ec="0.6", alpha=0.9))
    ax.legend(loc="upper right")
    return [_save(fig, "phonon_noise_vs_displacement.png", _rel(path))]


# --------------------------------------------------------------------------- #
# fig7 - elastic anisotropy
# --------------------------------------------------------------------------- #

def _elastic_rows(data: dict) -> dict[str, dict]:
    node = data.get("cells")
    if not isinstance(node, dict) and isinstance(data.get("records"), list):
        node = {r.get("cell") or f"{r.get('backend')}/{r.get('dtype')}": r
                for r in data["records"]}
    if not isinstance(node, dict):  # cells at the top level, keyed "backend/dtype"
        node = {k: v for k, v in data.items()
                if isinstance(v, dict) and "/" in k}
    rows = {}
    for cell, d in node.items():
        cij = {name: _first(d, f"{name}_Nm", f"{name}_Npm", f"{name}_N_per_m",
                            f"{name}_2D_Npm", name, name.lower())
               for name in ("C11", "C22", "C12")}
        if cij["C11"] is not None and cij["C22"] is not None:
            rows[cell] = {k: (float(v) if v is not None else None)
                          for k, v in cij.items()}
    return rows


def fig_elastic() -> list[Path]:
    path = _first_existing(RAW / "elastic.json", RAW / "elastic" / "elastic.json")
    rows = _elastic_rows(json.loads(path.read_text()))
    if not rows:
        raise FileNotFoundError(f"C11/C22 entries in {_rel(path)}")

    cells = sorted(rows, key=lambda c: (c != "e3nn/float64", c))
    x = np.arange(len(cells))
    comps = ("C11", "C22", "C12")
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for j, comp in enumerate(comps):
        vals = [rows[c][comp] if rows[c][comp] is not None else np.nan
                for c in cells]
        bars = ax.bar(x + (j - 1) * 0.27, vals, 0.25, color=OKABE_ITO[j],
                      label=f"{comp} (N/m)")
        ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=2)
    ax.axhline(LIT_C11_NPM, color=OKABE_ITO[0], ls="--", lw=1.1,
               label=f"DFT C11 ~{LIT_C11_NPM:.0f}")
    ax.axhline(LIT_C22_NPM, color=OKABE_ITO[1], ls="--", lw=1.1,
               label=f"DFT C22 ~{LIT_C22_NPM:.0f}")
    top = max(v for c in cells for v in
              [rows[c]["C11"], rows[c]["C22"], rows[c]["C12"] or 0.0])
    for i, c in enumerate(cells):
        ax.text(i, max(top, LIT_C22_NPM) * 1.06,
                f"C22/C11 = {rows[c]['C22'] / rows[c]['C11']:.1f}",
                ha="center", fontsize=7.5)
    ax.set_ylim(top=max(top, LIT_C22_NPM) * 1.18)
    ax.set_xticks(x, cells)
    ax.set_ylabel("2D elastic constant (N/m)")
    ax.set_title("All cells must keep C22 >> C11 (DFT anisotropy ~4) before "
                 "trusting strained MD")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    return [_save(fig, "elastic_anisotropy.png", _rel(path))]


# --------------------------------------------------------------------------- #
# fig8 - MD traces: NVE drift + NPT lattice
# --------------------------------------------------------------------------- #

def _md_traces(kind: str) -> list[tuple[str, Path, np.lib.npyio.NpzFile]]:
    md_dir = RAW / "md"
    files = [f for f in sorted(md_dir.glob("*.npz")) if kind in f.name.lower()]
    out = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        backend, dtype = _npz_text(z, "backend"), _npz_text(z, "dtype")
        cell = (f"{backend}/{dtype}" if backend and dtype
                else _npz_text(z, "cell") or _parse_cell_tag(f.name) or f.stem)
        out.append((cell, f, z))
    return out


def _time_ps(z) -> np.ndarray:
    for keys, scale in ((("time_ps", "t_ps"), 1.0), (("time_fs", "t_fs"), 1e-3)):
        try:
            return _npz_pick(z, *keys) * scale
        except KeyError:
            pass
    raise KeyError(f"no time array (time_ps/t_ps/time_fs/t_fs) in npz "
                   f"(has {z.files})")


def _make_nve(traces) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for cell, f, z in traces:
        t = _time_ps(z)
        try:
            e = _npz_pick(z, "etot_per_atom_eV", "etot_eV_per_atom",
                          "e_per_atom_eV")
        except KeyError:
            etot = _npz_pick(z, "etot_eV", "e_total_eV", "total_energy_eV")
            n = _npz_scalar(z, "natoms")
            if n is None:
                raise KeyError(f"{f.name}: total energy without natoms")
            e = etot / n
        slope_ue = np.polyfit(t, e, 1)[0] * 1e6  # microeV/atom/ps
        ax.plot(t, (e - e[0]) * 1e3, lw=0.9, color=_cell_color(cell),
                label=f"{cell}  {slope_ue:+.2f} $\\mu$eV/atom/ps")
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("$E_{tot}/N - E_0$ (meV/atom)")
    ax.set_title("Accept a cell for long MD only if NVE drift stays at the "
                 "$\\mu$eV/atom/ps level")
    ax.legend(title="cell  (drift slope)")
    return _save(fig, "nve_drift.png", "md/nve_*.npz")


def _make_npt(traces) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    for axis, key_names, lit in zip(
            axes, (("a_A", "a_super_A", "a_angstrom", "a"),
                   ("b_A", "b_super_A", "b_angstrom", "b")),
            (4.62, 3.30)):
        ref_tail = None
        for cell, f, z in traces:
            t = _time_ps(z)
            v = _npz_pick(z, *key_names)
            # tolerate supercell-length traces: fold back to per-unit-cell
            rep = max(1, round(float(np.median(v)) / lit))
            v = v / rep
            axis.plot(t, v, lw=0.9, color=_cell_color(cell), label=cell)
            if cell == "e3nn/float64":
                ref_tail = v[len(v) // 2:]  # equilibrated half
        if ref_tail is not None and len(ref_tail) > 1:
            m, s = float(np.mean(ref_tail)), float(np.std(ref_tail))
            axis.axhspan(m - s, m + s, color=_cell_color("e3nn/float64"),
                         alpha=0.15, label="fp64 ref $\\pm 1\\sigma$")
        axis.set_ylabel(f"{key_names[-1]} (A/unit cell)")
        axis.legend(ncol=2)
    axes[0].set_title("Trust fp32 thermal expansion only if a(t), b(t) stay "
                      "inside the fp64 reference band")
    axes[1].set_xlabel("time (ps)")
    return _save(fig, "npt_lattice.png", "md/npt_*.npz")


def fig_md_traces() -> list[Path]:
    if not (RAW / "md").is_dir():
        raise FileNotFoundError("results/raw/md/*.npz")
    written = []
    for kind, maker, name in (("nve", _make_nve, "nve_drift.png"),
                              ("npt", _make_npt, "npt_lattice.png")):
        traces = _md_traces(kind)
        if traces:
            written.append(maker(traces))
        else:
            print(f"SKIP {name}: missing results/raw/md/{kind}_*.npz",
                  flush=True)
    if not written:
        raise FileNotFoundError("results/raw/md/{nve,npt}_*.npz")
    return written


# --------------------------------------------------------------------------- #
# fig9 - nsys time share
# --------------------------------------------------------------------------- #

def _timeshare_rows(data) -> list[dict]:
    recs = data.get("records") if isinstance(data, dict) else data
    if recs is None and isinstance(data, dict) \
            and isinstance(data.get("traces"), dict):
        # canonical 31_parse_nsys.py schema: traces: {name: {..., time_share}}
        recs = [{**t, **(t.get("time_share") or {})}
                for t in data["traces"].values()
                if isinstance(t, dict) and "error" not in t]
    if not isinstance(recs, list):
        return []
    rows = []
    for r in recs:
        c = r.get("config", {}) if isinstance(r.get("config"), dict) else {}
        backend = r.get("backend") or c.get("backend")
        dtype = r.get("dtype") or c.get("dtype") or ""
        natoms = r.get("natoms") or c.get("natoms")
        kern = _first(r, "kernel_pct", "kernel_share", "kernel_s", "kernel")
        mcpy = _first(r, "memcpy_pct", "memcpy_share", "memcpy_s", "memcpy",
                      "transfer_pct") or 0.0
        host = _first(r, "host_other_pct", "host_pct", "other_pct",
                      "host_other_s", "host_s", "other_s", "host_other",
                      "other")
        if backend is None or natoms is None or kern is None:
            continue
        kern, mcpy = float(kern), float(mcpy)
        if host is None:  # percent- or fraction-style records may omit it
            total = 1.0 if kern + mcpy <= 1.001 else \
                100.0 if kern + mcpy <= 100.1 else None
            if total is None:
                continue
            host = total - kern - mcpy
        parts = np.array([kern, mcpy, float(host)], float)
        rows.append({"backend": backend, "dtype": dtype, "natoms": int(natoms),
                     "shares": parts / parts.sum()})
    return rows


def fig_timeshare() -> list[Path]:
    path = _first_existing(RAW / "nsys_summary.json",
                           RAW / "nsys" / "nsys_summary.json")
    rows = _timeshare_rows(json.loads(path.read_text()))
    if not rows:
        raise FileNotFoundError(f"kernel/memcpy/host records in {_rel(path)}")

    rows.sort(key=lambda r: (r["natoms"], r["backend"], r["dtype"]))
    short = {"float32": "fp32", "float64": "fp64"}
    labels = [f"{r['natoms']:,}\n{r['backend']}"
              + (f"/{short.get(r['dtype'], r['dtype'])}" if r["dtype"] else "")
              for r in rows]
    x = np.arange(len(rows))
    comps = ("GPU kernels", "memcpy H2D/D2H", "host + other")
    colors = (OKABE_ITO[0], OKABE_ITO[1], "0.65")

    fig, ax = plt.subplots(figsize=(max(6.0, 1.0 + 0.85 * len(rows)), 4.4))
    bottom = np.zeros(len(rows))
    for j, (comp, color) in enumerate(zip(comps, colors)):
        vals = np.array([r["shares"][j] for r in rows])
        ax.bar(x, vals, 0.65, bottom=bottom, color=color, label=comp)
        bottom += vals
    for i, r in enumerate(rows):
        ax.text(i, r["shares"][0] / 2, f"{100 * r['shares'][0]:.0f}%",
                ha="center", va="center", fontsize=7, color="w")
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("share of step time")
    ax.set_title("Faster kernels cannot fix host-bound small systems -\n"
                 "this is why break-even sits where it does")
    ax.legend(loc="upper right", framealpha=0.95)
    return [_save(fig, "time_share.png", _rel(path))]


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

@dataclass
class FigSpec:
    fig_id: str
    outputs: tuple[str, ...]
    builder: Callable[[], list[Path]]


SPECS = [
    FigSpec("fig1", ("throughput_vs_size.png",), fig_throughput),
    FigSpec("fig2", ("speedup_breakeven.png",), fig_breakeven),
    FigSpec("fig3", ("vram_oom.png",), fig_vram),
    FigSpec("fig4", ("parity_gates.png",), fig_parity),
    FigSpec("fig5", ("phonon_dispersion.png",), fig_phonon_dispersion),
    FigSpec("fig6", ("phonon_noise_vs_displacement.png",), fig_phonon_noise),
    FigSpec("fig7", ("elastic_anisotropy.png",), fig_elastic),
    FigSpec("fig8", ("nve_drift.png", "npt_lattice.png"), fig_md_traces),
    FigSpec("fig9", ("time_share.png",), fig_timeshare),
]


def _matches(spec: FigSpec, token: str) -> bool:
    return token == spec.fig_id or any(
        token in (o, Path(o).stem) for o in spec.outputs)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", default=None,
                   help="comma list of figures: fig1..fig9 or output stems "
                        "(e.g. 'fig1,phonon_dispersion')")
    p.add_argument("--smoke", action="store_true",
                   help="accepted for queue uniformity; plotting is already <2 min")
    args = p.parse_args()
    _setup_style()

    specs = SPECS
    if args.only:
        tokens = [t.strip() for t in args.only.split(",") if t.strip()]
        specs = [s for s in SPECS if any(_matches(s, t) for t in tokens)]
        unknown = [t for t in tokens
                   if not any(_matches(s, t) for s in SPECS)]
        if unknown:
            print(f"[phosbench] unknown --only tokens ignored: {unknown}")
        if not specs:
            print("[phosbench] --only matched no figures")
            return 2
    if args.smoke:
        print("[phosbench] --smoke: no-op (figure rendering is already cheap)")

    failures = 0
    for spec in specs:
        print(f"--- {spec.fig_id}: {' + '.join(spec.outputs)}", flush=True)
        try:
            spec.builder()
        except FileNotFoundError as exc:
            print(f"SKIP {spec.outputs[0]}: missing {exc}", flush=True)
        except Exception:
            failures += 1
            tb = traceback.format_exc()
            print(f"ERROR {spec.outputs[0]}: {tb.strip().splitlines()[-1]}")
            print(tb[-1500:], flush=True)

    print("\nfigure inventory (results/figures/):")
    for name in [o for s in SPECS for o in s.outputs] + ["oom_boundary.csv"]:
        mark = "x" if (FIGS / name).exists() else " "
        print(f"  [{mark}] {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
