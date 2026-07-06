#!/usr/bin/env bash
# Stage C - nsys traces of representative sweep configs (kernel-truth + time-share).
#
# Wraps the existing sweep child mode (scripts/10_sweep_throughput.py --single)
# in `nsys profile` so the profiled loop is bit-identical to what Stage B times;
# the NVTX `force_eval` ranges emitted by phosbench.common let 31_parse_nsys.py
# split wall time into kernel / memcpy / host shares.
#
# Config plan per size (nx,ny):
#   e3nn/float32 and cueq/float32 with the chosen foundation model; PLUS
#   e3nn/float64 with plain 'medium' (the accuracy-reference column) at the
#   SMALLEST requested size only - GA102 runs fp64 at 1:64 of fp32, and one
#   small trace already answers "what does the fp64 timeline look like".
#
# Usage:
#   bash scripts/30_profile_nsys.sh [--sizes "5,7 23,32"] [--model TAG]
#                                   [--steps 30] [--smoke]
#   --model default: Stage A winner from configs/model_choice.json if present,
#           else medium-mpa-0.
#   --smoke: one 4x4 cueq/float32 trace, 5 steps (<2 min with cached weights) -
#            still exercises the kernel-truth gate end to end.
#
# Per-run failures are tolerated and logged (results/raw/nsys/errors.log). The
# ncu permission probe never fails the script (PROTOCOL Stage A gate 5: note
# whether counters need the NVreg flag, don't block on it). Final exit code
# comes from 31_parse_nsys.py: 0 ok, 1 kernel-truth gate failed, 2 nothing
# parseable.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${PYTHON:-python}"
NSYS_DIR="$REPO/results/raw/nsys"
ERRLOG="$NSYS_DIR/errors.log"
RUN_TIMEOUT="${PHOSBENCH_NSYS_TIMEOUT_S:-1800}"

SIZES="5,7 23,32"
MODEL=""
STEPS=30
SMOKE=0

usage() {
    sed -n '2,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sizes)  SIZES="$2"; shift 2 ;;
        --model)  MODEL="$2"; shift 2 ;;
        --steps)  STEPS="$2"; shift 2 ;;
        --smoke)  SMOKE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[30_profile] unknown arg: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if ! command -v nsys >/dev/null 2>&1; then
    echo "[30_profile] FATAL: nsys not on PATH" >&2
    exit 2
fi

# Stage A's physics gate writes the winning foundation model to
# configs/model_choice.json; if no model passed, use the documented zero-shot
# baseline so Stage C stays aligned with the README narrative.
if [[ -z "$MODEL" ]]; then
    MODEL="$("$PY" -c '
import json
from pathlib import Path
p = Path("configs/model_choice.json")
d = json.loads(p.read_text()) if p.exists() else {}
print(d.get("model") or d.get("winner") or d.get("choice") or "medium-omat-0")
' 2>/dev/null || true)"
    MODEL="${MODEL:-medium-omat-0}"
fi

if [[ "$SMOKE" == 1 ]]; then
    SIZES="4,4"
    STEPS=5
fi

for sz in $SIZES; do
    if ! [[ "$sz" =~ ^[0-9]+,[0-9]+$ ]]; then
        echo "[30_profile] FATAL: bad --sizes token '$sz' (expected NX,NY)" >&2
        exit 2
    fi
done

# Smallest size (by atom count) hosts the lone fp64 reference trace.
SMALL=""
small_atoms=0
for sz in $SIZES; do
    at=$((4 * ${sz%,*} * ${sz#*,}))
    if [[ -z "$SMALL" || $at -lt $small_atoms ]]; then
        SMALL="$sz"
        small_atoms=$at
    fi
done

# coreutils timeout guards against a hung target; absent (e.g. macOS), run bare.
with_timeout() {
    if command -v timeout >/dev/null 2>&1; then
        timeout "$RUN_TIMEOUT" "$@"
    else
        "$@"
    fi
}

mkdir -p "$NSYS_DIR"
N_OK=0
N_FAIL=0

echo "[30_profile] model=$MODEL sizes='$SIZES' steps=$STEPS smoke=$SMOKE"
echo "[30_profile] traces -> $NSYS_DIR"

run_trace() {
    local backend="$1" dtype="$2" model="$3" nx="$4" ny="$5"
    local tag="${model}_${backend}_${dtype}_${nx}x${ny}"
    local log="$NSYS_DIR/${tag}.log"
    local rep="$NSYS_DIR/${tag}.nsys-rep"
    local rc=0
    echo "[30_profile] TRACE $tag (force_call, $STEPS steps)"
    with_timeout nsys profile --trace=cuda,nvtx,osrt --force-overwrite true \
        -o "$NSYS_DIR/$tag" \
        "$PY" scripts/10_sweep_throughput.py \
        --single "$backend" "$dtype" "$model" "$nx" "$ny" force_call \
        --n-steps "$STEPS" >"$log" 2>&1 || rc=$?
    if [[ -f "$rep" ]]; then
        N_OK=$((N_OK + 1))
        if [[ $rc -ne 0 ]]; then
            # nsys propagates the target's exit code; a crashed target can
            # still leave a usable partial timeline - keep it, flag it.
            echo "[30_profile]   WARN rc=$rc but trace written, keeping $rep" \
                | tee -a "$ERRLOG"
        else
            echo "[30_profile]   ok -> $rep"
        fi
    else
        N_FAIL=$((N_FAIL + 1))
        {
            echo "[30_profile]   FAIL rc=$rc tag=$tag (full log: $log)"
            tail -n 5 "$log" 2>/dev/null | sed 's/^/[30_profile]     | /'
        } | tee -a "$ERRLOG"
    fi
    return 0
}

for sz in $SIZES; do
    nx="${sz%,*}"
    ny="${sz#*,}"
    if [[ "$SMOKE" == 1 ]]; then
        run_trace cueq float32 "$MODEL" "$nx" "$ny"
    else
        run_trace e3nn float32 "$MODEL" "$nx" "$ny"
        run_trace cueq float32 "$MODEL" "$nx" "$ny"
    fi
done

# fp64 accuracy-reference timeline: plain 'medium', smallest size only.
if [[ "$SMOKE" != 1 ]]; then
    run_trace e3nn float64 medium "${SMALL%,*}" "${SMALL#*,}"
fi

# --- ncu permission probe (Stage A gate 5: note, never block) ---------------
PROBE_TXT="$REPO/results/raw/ncu_probe.txt"
PROBE_LOG="$NSYS_DIR/ncu_probe.log"
echo "[30_profile] ncu permission probe (hardware counters often need NVreg flag)"
rm -f /tmp/ncu_probe.ncu-rep
if command -v ncu >/dev/null 2>&1 \
    && ncu --launch-count 1 --kill yes -o /tmp/ncu_probe \
        "$PY" -c 'import torch; (torch.zeros(1024,1024,device="cuda")@torch.zeros(1024,1024,device="cuda")).sum().item()' \
        >"$PROBE_LOG" 2>&1; then
    {
        echo "ncu probe OK $(date -Iseconds) - hardware counters available"
        echo "report: /tmp/ncu_probe.ncu-rep (deep-dive on dominant kernel is unblocked)"
    } >"$PROBE_TXT"
    echo "[30_profile]   ncu probe OK"
else
    {
        echo "ERR_NVGPUCTRPERM or unavailable"
        echo "--- probe output tail ($(date -Iseconds)) ---"
        tail -n 20 "$PROBE_LOG" 2>/dev/null || echo "(ncu not on PATH)"
    } >"$PROBE_TXT"
    echo "[30_profile]   ncu counters unavailable (recorded in $PROBE_TXT, not fatal)"
fi

# --- reduce traces to results/raw/nsys_summary.json --------------------------
if [[ $N_OK -eq 0 ]]; then
    echo "[30_profile] ERROR: no trace succeeded this run (failures: $N_FAIL," \
        "see $ERRLOG)" >&2
fi
echo "[30_profile] traces ok=$N_OK fail=$N_FAIL; parsing with 31_parse_nsys.py"
"$PY" scripts/31_parse_nsys.py
exit $?
