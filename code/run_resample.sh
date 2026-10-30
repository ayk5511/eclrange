#!/usr/bin/env bash
# Post-hoc independent-sample refits, three dates at a time.
set -uo pipefail
cd "$(dirname "$0")"
export ECL_THREADS=6 OMP_NUM_THREADS=6
printf "%s\n" "$@" | xargs -P 3 -I{} bash -c 'python3 04b_resample.py {} > ../logs/resample_{}.log 2>&1; echo "{} resample exit $?"'
