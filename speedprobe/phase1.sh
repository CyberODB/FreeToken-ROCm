#!/bin/bash
# Phase 1 A/Bs (doc 18), 2026-11-06. Base: ratio 0.85, pin 14, cpu-layers auto, cache 5000. Tests decode + 7K prompt.
cd /e/freetoken-021/scripts
R="--memory-ratio 0.85"
T="--tests decode,p5k"
run() { echo "=== $1"; py -u ft-speedprobe.py "$@" $T 2>&1 | grep -v loading | grep -E "moe-cpu-layers|pinned|Allocating|Free memory after|run \.\.\.|ERROR" | cut -c1-260; }
#run p1-base     --args "$R"
#run p1-pinall   --env FREETOKEN_PIN_BUDGET_GB= --args "$R"
run p1-thr6     --args "$R --moe-cpu-threads 6"
run p1-thr8     --args "$R --moe-cpu-threads 8"
run p1-mrr1     --args "$R --max-running-requests 1"
run p1-textonly --args "$R --text-model-only"
run p1-hybrid   --args "$R --moe-strategy hybrid"
echo "=== done"
