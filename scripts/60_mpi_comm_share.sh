#!/usr/bin/env bash
# B-1 - MPI communication-share methodology demo (classical-potential CPU).
#
# WHAT THIS IS: an SA-style profiling demo. It runs a fixed-size LJ-melt
# benchmark under LAMMPS+MPI at rank counts np in {1,2,4,8} (strong scaling on
# one node) and reads LAMMPS's own loop-timing breakdown (Pair/Neigh/Comm/...).
# The point is the METHODOLOGY: how you tell whether a customer workload is
# comm-bound, and at what rank count communication starts eating the pair-time
# win.
#
# WHAT THIS IS NOT: this is NOT MACE-in-LAMMPS. It is the classical Lennard-Jones
# pair potential. The comm/compute breakdown LAMMPS prints is the SAME table you
# read for an ML-IAP (MACE) run at scale (via LAMMPS ML-IAP / pair_style mace),
# so the reading transfers - but every number here is classical LJ, single node,
# one run per rank count.
#
# Runs on the workstation (Ryzen 5800X: 8 physical cores / 16 threads). We pin to
# PHYSICAL cores only (np <= 8) and do NOT use SMT - hyperthreads share FPUs and
# would muddy a compute/comm attribution. One run per np (this is a methodology
# demo, not a statistics campaign).
#
# Usage:
#   bash scripts/60_mpi_comm_share.sh [--env NAME] [--atoms N] [--steps N]
#                                     [--np "1 2 4 8"] [--smoke]
#   --env    conda env with lammps+openmpi (default: lammpsmpi)
#   --atoms  target atom count, rounded to the LJ fcc lattice (default: 500000)
#   --steps  MD steps per run (default: 800)
#   --np     space-separated rank counts (default: "1 2 4 8")
#   --smoke  tiny/fast: 32000 atoms, 100 steps, np "1 2" (end-to-end sanity)
#
# Raw logs -> results/raw/mpi/lj_np<K>.log (gitignored). A per-run failure is
# logged and tolerated; the parser (61_parse_lammps_log.py) reduces whatever
# logs exist. Exit code comes from the parser: 0 ok, 2 nothing parseable.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

ENV_NAME="${CONDA_ENV:-lammpsmpi}"   # env-name parameterized like 51/stage_a
ATOMS=500000
STEPS=800
NPS="1 2 4 8"
SMOKE=0

usage() { sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)   ENV_NAME="$2"; shift 2 ;;
        --atoms) ATOMS="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --np)    NPS="$2"; shift 2 ;;
        --smoke) SMOKE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[60_mpi] unknown arg: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$SMOKE" == 1 ]]; then
    ATOMS=32000
    STEPS=100
    NPS="1 2"
fi

MPI_DIR="$REPO/results/raw/mpi"
mkdir -p "$MPI_DIR"

# --- resolve conda + binaries -------------------------------------------------
for CONDA_SH in ~/miniforge3/etc/profile.d/conda.sh \
                ~/miniconda3/etc/profile.d/conda.sh \
                /opt/conda/etc/profile.d/conda.sh; do
    # shellcheck disable=SC1090
    [[ -f "$CONDA_SH" ]] && { source "$CONDA_SH"; break; }
done
if command -v conda >/dev/null 2>&1; then
    conda activate "$ENV_NAME" || {
        echo "[60_mpi] FATAL: cannot activate env '$ENV_NAME'" >&2; exit 2; }
fi

# LAMMPS binary is 'lmp' on conda-forge, 'lmp_mpi' on some builds.
LMP=""
for cand in lmp lmp_mpi; do
    command -v "$cand" >/dev/null 2>&1 && { LMP="$cand"; break; }
done
if [[ -z "$LMP" ]]; then
    echo "[60_mpi] FATAL: no lmp / lmp_mpi on PATH (env '$ENV_NAME')" >&2; exit 2
fi
if ! command -v mpirun >/dev/null 2>&1; then
    echo "[60_mpi] FATAL: mpirun not on PATH (env '$ENV_NAME')" >&2; exit 2
fi
echo "[60_mpi] env=$ENV_NAME lmp=$(command -v "$LMP") mpirun=$(command -v mpirun)"

# --- LJ-melt lattice sizing ---------------------------------------------------
# 'region ... units lattice' + 'create_atoms 1 box' on an fcc lattice yields
# 4 * L^3 atoms for an L x L x L box. Pick L so 4*L^3 ~ ATOMS.
L=$(python3 - "$ATOMS" <<'PY'
import sys
atoms = int(sys.argv[1])
L = round((atoms / 4.0) ** (1.0 / 3.0))
print(max(1, int(L)))
PY
)
NATOMS=$((4 * L * L * L))
echo "[60_mpi] LJ fcc box: L=$L -> $NATOMS atoms (target $ATOMS), steps=$STEPS"
echo "[60_mpi] ranks: $NPS (physical cores only; SMT NOT used)"

# --- LAMMPS input (written here so the harness is self-contained) -------------
# Textbook LJ melt (bench/in.lj style): fcc lattice, LJ cut 2.5, velocity seed,
# NVE, run STEPS. Same physics at every np - only the domain decomposition (and
# therefore the Comm share) changes with rank count. This is STRONG scaling:
# fixed total atoms, more ranks.
IN="$MPI_DIR/in.lj"
cat > "$IN" <<EOF
# LJ melt strong-scaling benchmark (B-1 MPI comm-share demo). Fixed size.
variable        L equal $L
variable        steps equal $STEPS

units           lj
atom_style      atomic

lattice         fcc 0.8442
region          box block 0 \${L} 0 \${L} 0 \${L}
create_box      1 box
create_atoms    1 box
mass            1 1.0

velocity        all create 1.44 87287 loop geom

pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5

neighbor        0.3 bin
neigh_modify    delay 0 every 20 check no

fix             1 all nve

# quiet the run: no thermo I/O in the timed loop except endpoints
thermo          \${steps}
thermo_modify   norm no

run             \${steps}
EOF

# --- run each rank count ------------------------------------------------------
N_OK=0
for np in $NPS; do
    log="$MPI_DIR/lj_np${np}.log"
    echo "[60_mpi] RUN np=$np -> $(basename "$log")"
    # --bind-to core --map-by core: one rank per physical core, no oversubscribe.
    # --use-hwthread-cpus is deliberately OMITTED so we stay on physical cores.
    if mpirun -np "$np" --bind-to core --map-by core \
            "$LMP" -in "$IN" -log "$log" -screen none \
            >>"$MPI_DIR/run.out" 2>&1; then
        N_OK=$((N_OK + 1))
        echo "[60_mpi]   ok"
    else
        # retry without binding flags (some OpenMPI builds refuse --bind-to on
        # oversubscribed / container hosts); the timing table is still valid.
        echo "[60_mpi]   bind failed, retrying without --bind-to/--map-by" >&2
        if mpirun -np "$np" \
                "$LMP" -in "$IN" -log "$log" -screen none \
                >>"$MPI_DIR/run.out" 2>&1; then
            N_OK=$((N_OK + 1))
            echo "[60_mpi]   ok (unbound)"
        else
            echo "[60_mpi]   FAIL np=$np (see $MPI_DIR/run.out)" >&2
        fi
    fi
done

echo "[60_mpi] runs ok=$N_OK; parsing with 61_parse_lammps_log.py"
python3 scripts/61_parse_lammps_log.py --dir "$MPI_DIR"
exit $?
