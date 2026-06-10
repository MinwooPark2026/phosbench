#!/usr/bin/env python
"""Stage C - reduce nsys traces to a kernel / memcpy / host time-share summary.

For every results/raw/nsys/*.nsys-rep this runs

    nsys stats --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
        --report nvtx_sum --format csv --force-export=true <rep>

and parses the concatenated CSV sections (robustly, by section header - nsys
prints export notices and report banners between tables, and column sets vary
slightly across nsys versions). Per trace it extracts:

  - top-10 kernels (name, total time, % of kernel time),
  - total kernel time, total H2D / D2H memcpy time,
  - NVTX `force_eval` total (instrumented in phosbench.common),
  - cueq_kernels_present: any kernel name matching
    /cuequivariance|cueq|segmented_polynomial|segmented.*tensor|tensor_product/i (segmented_polynomial_* are the cuEq ops kernels observed on SM86).

cueq_kernels_present is the silent-fallback detector (Stage A kernel-truth
gate): enable_cueq=True must put cuEquivariance kernels on the GPU timeline,
otherwise MACE quietly ran e3nn and every "cueq" speedup number is fiction.

Time shares use the NVTX force_eval total as denominator - the wall time the
calculator was actually asked to work - so host% is genuine Python/ASE/launch
overhead inside force evaluations, not model-load time.

Exit codes: 0 ok; 1 kernel-truth gate failed (a cueq trace with no cuEq
kernels); 2 no traces found or none parseable.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phosbench.common import env_metadata, write_json

REPO = Path(__file__).resolve().parent.parent

SECTION_RE = re.compile(r"\*\*.*\((\w+)\)")
# nsys >= ~2024 drops the '** Title (key):' banners; the section opener is the
# per-report processing notice instead:
#   Processing [foo.sqlite] with [/opt/nvidia/.../reports/cuda_gpu_kern_sum.py]...
PROCESSING_RE = re.compile(r"Processing \[.*\] with \[.*[/\\](\w+)\.py\]")
CUEQ_RE = re.compile(r"cuequivariance|cueq|segmented_polynomial|segmented.*tensor|tensor_product", re.I)
H2D_RE = re.compile(r"htod|host-?to-?device", re.I)
D2H_RE = re.compile(r"dtoh|device-?to-?host", re.I)
TAG_RE = re.compile(
    r"(?P<model>.+)_(?P<backend>e3nn|cueq)_(?P<dtype>float\d+)_(?P<nx>\d+)x(?P<ny>\d+)$"
)


# --------------------------------------------------------------------------- #
# CSV-section parsing
# --------------------------------------------------------------------------- #

def split_sections(text: str) -> dict[str, list[str]]:
    """Map report key ('cuda_gpu_kern_sum', ...) -> raw lines of its section.

    Sections are introduced by ' ** <Title> (<report_key>):' banner lines;
    everything else (Processing/SKIPPED notices) lands outside any table and
    is filtered later by requiring a parseable 'Total Time' per row.
    """
    sections: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        m = SECTION_RE.search(line)
        if m and "**" in line:
            current = m.group(1)
            sections[current] = []
            continue
        m = PROCESSING_RE.search(line)
        if m:
            current = m.group(1)
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)
    return sections


def read_rows(lines: list[str]) -> list[dict]:
    """Parse one CSV table, tolerating padding and stray notices around it."""
    header_i = next(
        (i for i, ln in enumerate(lines) if "," in ln and "Time" in ln), None
    )
    if header_i is None:
        return []
    body = [ln for ln in lines[header_i:] if ln.strip()]
    rows = []
    for raw in csv.DictReader(io.StringIO("\n".join(body))):
        clean = {
            k.strip(): v.strip()
            for k, v in raw.items()
            if isinstance(k, str) and isinstance(v, str)
        }
        if _num(clean, "Total Time") is not None:
            rows.append(clean)
    return rows


def _num(row: dict, key_substr: str) -> float | None:
    """First column whose name contains key_substr, as float (else None)."""
    for k, v in row.items():
        if key_substr.lower() in k.lower():
            try:
                return float(v.replace(",", ""))
            except ValueError:
                return None
    return None


def _name(row: dict) -> str:
    for key in ("Name", "Operation", "Range"):
        if key in row:
            return row[key]
    return ""


# --------------------------------------------------------------------------- #
# Per-trace summary
# --------------------------------------------------------------------------- #

def parse_tag(stem: str) -> dict:
    """Recover config from the trace name <model>_<backend>_<dtype>_<nx>x<ny>."""
    m = TAG_RE.match(stem)
    if not m:
        return {"model": None, "backend": None, "dtype": None, "natoms": None}
    d = m.groupdict()
    nx, ny = int(d["nx"]), int(d["ny"])
    return {"model": d["model"], "backend": d["backend"], "dtype": d["dtype"],
            "nx": nx, "ny": ny, "natoms": 4 * nx * ny}


def summarize(rep: Path) -> dict:
    cmd = ["nsys", "stats",
           "--report", "cuda_gpu_kern_sum",
           "--report", "cuda_gpu_mem_time_sum",
           "--report", "nvtx_sum",
           "--format", "csv", "--force-export=true", str(rep)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0 and not proc.stdout.strip():
        raise RuntimeError(
            f"nsys stats rc={proc.returncode}: {(proc.stderr or '')[-500:]}"
        )
    sections = split_sections(proc.stdout)
    kern = read_rows(sections.get("cuda_gpu_kern_sum", []))
    mem = read_rows(sections.get("cuda_gpu_mem_time_sum", []))
    nvtx = read_rows(sections.get("nvtx_sum", []))

    kern = sorted(kern, key=lambda r: _num(r, "Total Time") or 0.0, reverse=True)
    kernel_total_ms = sum((_num(r, "Total Time") or 0.0) for r in kern) / 1e6
    top10 = [{"name": _name(r),
              "total_ms": (_num(r, "Total Time") or 0.0) / 1e6,
              "pct": _num(r, "Time (%)")} for r in kern[:10]]
    h2d_ms = sum((_num(r, "Total Time") or 0.0)
                 for r in mem if H2D_RE.search(_name(r))) / 1e6
    d2h_ms = sum((_num(r, "Total Time") or 0.0)
                 for r in mem if D2H_RE.search(_name(r))) / 1e6
    force_rows = [r for r in nvtx if "force_eval" in _name(r)]
    nvtx_force_ms = (
        sum((_num(r, "Total Time") or 0.0) for r in force_rows) / 1e6
        if force_rows else None
    )

    share = None
    if nvtx_force_ms:
        kernel_pct = 100.0 * kernel_total_ms / nvtx_force_ms
        memcpy_pct = 100.0 * (h2d_ms + d2h_ms) / nvtx_force_ms
        share = {"basis": "nvtx_force_eval",
                 "kernel_pct": kernel_pct,
                 "memcpy_pct": memcpy_pct,
                 "host_pct": max(0.0, 100.0 - kernel_pct - memcpy_pct)}

    return {
        "n_kernels": len(kern),
        "top_kernels": top10,
        "kernel_total_ms": kernel_total_ms,
        "memcpy_h2d_ms": h2d_ms,
        "memcpy_d2h_ms": d2h_ms,
        "nvtx_force_eval_ms": nvtx_force_ms,
        "cueq_kernels_present": any(CUEQ_RE.search(_name(r)) for r in kern),
        "time_share": share,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", default=str(REPO / "results/raw/nsys"))
    p.add_argument("--out", default=str(REPO / "results/raw/nsys_summary.json"))
    args = p.parse_args()

    reps = sorted(Path(args.dir).glob("*.nsys-rep"))
    if not reps:
        print(f"[31_parse] no .nsys-rep under {args.dir} - "
              "run scripts/30_profile_nsys.sh first", file=sys.stderr)
        return 2

    traces = {}
    for rep in reps:
        print(f"[31_parse] {rep.name}", flush=True)
        try:
            traces[rep.stem] = {**parse_tag(rep.stem), **summarize(rep)}
        except Exception as exc:  # one bad trace must not sink the rest
            print(f"[31_parse]   ERROR {exc}", file=sys.stderr, flush=True)
            traces[rep.stem] = {**parse_tag(rep.stem), "error": str(exc)}

    write_json(args.out, {"env": env_metadata(), "traces": traces})

    def fmt(x):
        return f"{x:6.1f}" if x is not None else "   n/a"

    print(f"\n{'trace':46s} {'kern%':>6s} {'mcpy%':>6s} {'host%':>6s} "
          f"{'cueq':>5s}  top kernel", flush=True)
    for name in sorted(traces):
        t = traces[name]
        if "error" in t:
            print(f"{name:46s}  ERROR {t['error'][:70]}", flush=True)
            continue
        sh = t["time_share"] or {}
        top = t["top_kernels"][0]["name"][:50] if t["top_kernels"] else "-"
        print(f"{name:46s} {fmt(sh.get('kernel_pct'))} {fmt(sh.get('memcpy_pct'))} "
              f"{fmt(sh.get('host_pct'))} "
              f"{'yes' if t['cueq_kernels_present'] else 'NO':>5s}  {top}",
              flush=True)

    parsed = [n for n, t in traces.items() if "error" not in t]
    if not parsed:
        print("[31_parse] all traces failed to parse", file=sys.stderr)
        return 2

    cueq_traces = [n for n in parsed if traces[n].get("backend") == "cueq"]
    gate_fail = [n for n in cueq_traces if not traces[n]["cueq_kernels_present"]]
    if gate_fail:
        print(f"KERNEL_TRUTH: FAIL - no cuEquivariance kernels in "
              f"{', '.join(gate_fail)} (silent e3nn fallback suspected)",
              flush=True)
        return 1
    if cueq_traces:
        print("KERNEL_TRUTH: PASS - cuEquivariance kernels on the GPU timeline "
              "in every cueq trace", flush=True)
    else:
        print("KERNEL_TRUTH: n/a - no cueq traces parsed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
