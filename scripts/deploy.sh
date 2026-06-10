#!/usr/bin/env bash
# Sync the repo to workstation (push) or fetch measurement results back (pull).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
case "${1:-push}" in
  push)
    rsync -a --info=stats1 --exclude '__pycache__' --exclude '.git' \
      --exclude 'results/' "$REPO/" workstation:phosbench/
    ;;
  pull)
    rsync -a --info=stats1 workstation:phosbench/results/ "$REPO/results/"
    ;;
  *)
    echo "usage: deploy.sh [push|pull]" >&2
    exit 2
    ;;
esac
