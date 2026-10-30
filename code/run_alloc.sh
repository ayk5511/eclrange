#!/usr/bin/env bash
# Rerun only the allowance step (models already fitted), three dates at a time.
set -uo pipefail
cd "$(dirname "$0")"
export ECL_THREADS=6 OMP_NUM_THREADS=6
printf "%s\n" "$@" | xargs -P 3 -I{} bash -c 'python3 05_allowance.py {} > ../logs/alloc_{}.log 2>&1; echo "{} alloc exit $?"'
