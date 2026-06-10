#!/usr/bin/env python
"""The error-budget figure: precision/backend error vs model error, per observable.

One log-scale bar chart of RELATIVE errors (% of the observable's reference
scale) with three tiers per observable:
  T1   |fp32 - fp64|, same model, e3nn          (precision cost)
  T1'  |cuEq - e3nn|, fp32                      (backend cost)
  T2   |model(fp64) - DFT/Raman literature|     (model error)
The deployment message the chart must carry: T1/T1' sit 1-3 orders of
magnitude below T2 - pay for a better model, not for more precision.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "results/raw"
OUT = REPO / "results/figures/error_budget.png"

# literature anchors (PBE / Raman; see README finding 7 and PROTOCOL Stage D)
LIT = {"C11_Nm": 24.0, "C22_Nm": 103.0, "a_armchair_A": 4.62,
       "raman_top_cm1": 467.0}
CM1_PER_THZ = 33.356


def main() -> int:
    el = json.load(open(RAW / "elastic_corrected.json"))["cells"]
    ph = json.load(open(RAW / "phonons_summary.json"))
    gate = json.load(open(REPO / "configs/model_choice.json"))

    bars = {}  # observable -> (T1, T1', T2) in % of reference scale

    # elastic constants (reference scale = literature value)
    for key, lit in (("C11_Nm", LIT["C11_Nm"]), ("C22_Nm", LIT["C22_Nm"])):
        f64, f32, cq = (el[c][key] for c in
                        ("e3nn/float64", "e3nn/float32", "cueq/float32"))
        bars[key.replace("_Nm", " (N/m)")] = (
            abs(f32 - f64) / lit * 100,
            abs(cq - f32) / lit * 100,
            abs(f64 - lit) / lit * 100,
        )

    # phonons (reference scale = top Raman-active mode)
    runs = {(r["cell"], r["displacement"]): r for r in ph["runs"]}
    rmse = ph["rmse_vs_reference"]
    t1 = rmse["e3nn/float32@d0.01"]["overall_rmse_cm1"]
    t1cq = rmse["cueq/float32@d0.01"]["overall_rmse_cm1"]
    max64 = runs[("e3nn/float64", 0.01)]["max_freq_THz"] * CM1_PER_THZ
    bars["phonon freq (cm$^{-1}$)"] = (
        t1 / LIT["raman_top_cm1"] * 100,
        abs(t1cq - t1) / LIT["raman_top_cm1"] * 100,
        abs(max64 - LIT["raman_top_cm1"]) / LIT["raman_top_cm1"] * 100,
    )

    # armchair lattice constant (fp32 effect not separately measured -> T1 n/a)
    a_model = gate["details"]["medium-omat-0"]["a_armchair_A"]
    bars["lattice a (armchair)"] = (
        None, None, abs(a_model - LIT["a_armchair_A"]) / LIT["a_armchair_A"] * 100,
    )

    labels = list(bars)
    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 5.5))
    tiers = [
        ("fp32 cost  |fp32$-$fp64|", [bars[k][0] for k in labels], "#0072B2"),
        ("backend cost  |cuEq$-$e3nn|", [bars[k][1] for k in labels], "#009E73"),
        ("model error  |model$-$literature|", [bars[k][2] for k in labels], "#D55E00"),
    ]
    for i, (lab, vals, color) in enumerate(tiers):
        xs = [xi + (i - 1) * w for xi, v in zip(x, vals) if v is not None]
        vs = [v for v in vals if v is not None]
        b = ax.bar(xs, vs, w, label=lab, color=color)
        for rect, v in zip(b, vs):
            ax.annotate(f"{v:.3g}%", (rect.get_x() + rect.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 200)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("relative error (% of literature scale)")
    ax.set_title("Pay for a better model, not for more precision -\n"
                 "precision/backend errors sit 1-3 orders below model error")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.text(0.99, 0.01, "source: elastic_corrected.json, phonons_summary.json,"
             " model_choice.json", ha="right", fontsize=7, color="gray")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"[41_error_budget] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
