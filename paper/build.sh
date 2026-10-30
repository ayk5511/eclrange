#!/usr/bin/env bash
# Build Paper 10. Regenerates every number, table and figure from results/
# first, so the PDF can never show a number the results do not produce.
set -euo pipefail
VERSION="v0.1"
OUT_NAME="Khan_2026_cecl_model_range_${VERSION}.pdf"
cd "$(dirname "$0")"
PYTHONPATH=../code python3 ../code/08_figures.py
PYTHONPATH=../code python3 ../code/09_paper.py
tectonic main.tex
mkdir -p submission-ssrn
cp main.pdf "submission-ssrn/${OUT_NAME}"
echo "Built:           paper/main.pdf"
echo "Versioned copy:  paper/submission-ssrn/${OUT_NAME}"
