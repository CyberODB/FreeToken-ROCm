#!/bin/bash
# MTP gate for DFlash (doc 18 Phase 2), 2026-11-07. Hybrid 0.85 control, then MTP k=3 with the on-disk sidecar.
cd /e/freetoken-021/scripts
H="--memory-ratio 0.85 --moe-strategy hybrid"
MTP="--speculative-algorithm mtp --speculative-num-steps 3 --speculative-draft-model-path T:/ai-models/gguf/local/mtp-Qwen3.6-35B-A3B-Q4_0.gguf"
run() { echo "=== $1"; py -u ft-speedprobe.py "$@" 2>&1 | grep -v loading | grep -E "Free memory after|run \.\.\.|ERROR|Error|speculative|accept" | cut -c1-260; }
run g-hybrid085 --args "$H" --tests decode,p5k
run g-mtp3      --args "$H $MTP" --tests decode,p5k
echo "=== done"
