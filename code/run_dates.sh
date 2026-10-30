#!/usr/bin/env bash
# Fit models and compute allowances for every reporting date, three dates at
# a time with six threads each (18 cores). Logs go to logs/date_T.log.
set -uo pipefail
cd "$(dirname "$0")"
export ECL_THREADS=6 OMP_NUM_THREADS=6
run() {
  t=$1
  { python3 04_models.py "$t" && python3 05_allowance.py "$t"; } > "../logs/date_$t.log" 2>&1
  echo "$t exit $?"
}
export -f run
printf "%s\n" "$@" | xargs -P 3 -I{} bash -c 'run {}'
