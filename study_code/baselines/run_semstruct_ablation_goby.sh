#!/usr/bin/env bash
# SEMSTRUCT_ABLATION.md, Amendment 1, corpus 2: GOBY (independently produced). Training on GOBY's source tables only
# (self-supervised, as the paper), then the same arms as on Valentine. Offline: the e5 model is cached locally.
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=.venv-embed/Scripts/python.exe
R=reports/semstruct_ablation_20261027/goby
C=generated/goby-imported
mkdir -p "$R/full" "$R/onehop"
$PY -I semstruct_run.py --out "$R/full" --cases-root "$C" --delta-init zero --mining semihard train --threads 4 --tag a1 \
    > "$R/full/train_a1.log" 2>&1
$PY -I semstruct_run.py --out "$R/onehop" --cases-root "$C" --delta-init zero --mining semihard --onehop train --threads 4 \
    --tag a1 > "$R/onehop/train_a1.log" 2>&1
$PY -I semstruct_run.py --out "$R/full" --cases-root "$C" --delta-init zero --mining semihard score --regime plain --tag a1 \
    --arms semstruct lm_only permuted zero_delta --threads 4 > "$R/full/score_plain.log" 2>&1
$PY -I semstruct_run.py --out "$R/onehop" --cases-root "$C" --delta-init zero --mining semihard --onehop score --regime plain \
    --tag a1 --arms semstruct permuted --threads 4 > "$R/onehop/score_plain.log" 2>&1
echo "GOBY pipeline done"
