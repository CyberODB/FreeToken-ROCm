#!/bin/bash
# Phase 1 controls, 2026-11-06 23:2x (after the user moved video streams to CPU decode).
cd /e/freetoken-021/scripts
run() { echo "=== $1"; py -u ft-speedprobe.py "$@" 2>&1 | grep -v loading | grep -E "Free memory after|run \.\.\.|ERROR" | cut -c1-260; }
run c-base085   --args "--memory-ratio 0.85" --tests decode,p5k,p25k
run c-hybrid085 --args "--memory-ratio 0.85 --moe-strategy hybrid" --tests decode,p5k,p25k
run c-base090   --tests decode,p3k
run c-hybrid090 --args "--moe-strategy hybrid" --tests decode,p3k
echo "=== done"
