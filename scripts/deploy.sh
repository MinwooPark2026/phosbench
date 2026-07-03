#!/usr/bin/env bash
# Sync the repo to the workstation (push) or fetch measurement results back (pull).
set -euo pipefail
REMOTE="${PHOSBENCH_REMOTE:-workstation}"  # set PHOSBENCH_REMOTE to your ssh host
REPO="$(cd "$(dirname "$0")/.." && pwd)"
case "${1:-push}" in
  push)
    rsync -a --info=stats1 --exclude '__pycache__' --exclude '.git' \
      --exclude 'results/' "$REPO/" "$REMOTE":phosbench/
    ;;
  pull)
    rsync -a --info=stats1 "$REMOTE":phosbench/results/ "$REPO/results/"
    ;;
  *)
    echo "usage: deploy.sh [push|pull]" >&2
    exit 2
    ;;
esac
