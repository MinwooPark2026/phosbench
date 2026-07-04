#!/usr/bin/env python
"""Before/after figure for the model-fix arc: zero-shot vs GAP-20 fine-tuned.

Signed deviation from DFT/Raman literature per observable. The story in one
frame: the zero-shot failure lives on the soft armchair axis; ~2 h of
energy-weighted fine-tuning (training wall-time) brings every observable within ~5 %.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/figures/model_fix.png"
LIT = {"a (armchair)": 4.62, "b (zigzag)": 3.30,
       "C11": 24.0, "C22": 103.0, "top optical": 467.0}
CM1 = 33.356


def main() -> int:
    gate = json.load(open(REPO / "configs/model_choice.json"))
    zs = gate["details"]["medium-omat-0"]
    el = json.load(open(REPO / "results/raw/elastic_corrected.json"))["cells"]
    ph = json.load(open(REPO / "results/raw/phonons_summary.json"))
    ft = json.load(open(REPO / "results/raw/ft_validation.json"))

    zs_top = max(r["max_freq_THz"] for r in ph["runs"]
                 if r["cell"] == "e3nn/float64") * CM1
    before = {
        "a (armchair)": zs["a_armchair_A"],
        "b (zigzag)": zs["b_zigzag_A"],
        "C11": el["e3nn/float64"]["C11_Nm"],
        "C22": el["e3nn/float64"]["C22_Nm"],
        "top optical": zs_top,
    }
    after = {
        "a (armchair)": ft["lattice"]["a_A"],
        "b (zigzag)": ft["lattice"]["b_A"],
        "C11": ft["cells"]["e3nn/float64"]["C11"],
        "C22": ft["cells"]["e3nn/float64"]["C22"],
        "top optical": ft["cells"]["e3nn/float64"]["max_freq_THz"] * CM1,
    }

    labels = list(LIT)
    dev_b = [100 * (before[k] / LIT[k] - 1) for k in labels]
    dev_a = [100 * (after[k] / LIT[k] - 1) for k in labels]
    x = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, dev_b, w, label="zero-shot MACE-OMAT-0", color="#D55E00")
    ax.bar(x + w / 2, dev_a, w,
           label="fine-tuned on GAP-20 (~2 h fine-tune)", color="#009E73")
    for xi, (vb, va) in enumerate(zip(dev_b, dev_a)):
        ax.annotate(f"{vb:+.1f}%", (xi - w / 2, vb), ha="center",
                    va="bottom" if vb >= 0 else "top", fontsize=9)
        ax.annotate(f"{va:+.1f}%", (xi + w / 2, va), ha="center",
                    va="bottom" if va >= 0 else "top", fontsize=9)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhspan(-5, 5, color="gray", alpha=0.12, label="±5 % of literature")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("deviation from DFT/Raman literature (%)")
    ax.set_title("Validate, then fine-tune: a ~2 h fine-tune repairs the soft axis\n"
                 "(found by the geometry gate, fixed with energy-weighted "
                 "GAP-20 fine-tuning)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    # source watermark in a reserved bottom strip, outside the axes
    fig.text(0.006, 0.008, "source: model_choice.json, elastic_corrected.json, "
             "phonons_summary.json, ft_validation.json",
             ha="left", va="bottom", fontsize=6.5, color="0.55")
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 1.0))
    fig.savefig(OUT, dpi=200)
    print(f"[42_model_fix] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
