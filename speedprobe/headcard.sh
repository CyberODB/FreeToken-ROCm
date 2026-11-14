#!/bin/bash
# First load of merged 021 (PR #857) with a card's head args, FT-Serve base flags. 2026-11-07.
cd /e/freetoken-021/scripts
FS="--moe-cpu-layers auto --moe-strategy hybrid --memory-ratio 0.85"
CARD="--max-seq-len-override 65536 --speculative-algorithm mtp --speculative-num-steps 3 --speculative-draft-model-path T:/ai-models/gguf/local/mtp-Qwen3.6-35B-A3B-Q4_0.gguf"
DF="--max-seq-len-override 65536 --speculative-algorithm dflash --speculative-num-steps 3 --speculative-draft-model-path E:/ai-models/ft-models/incoai_Qwen3.6-35B-A3B-DFlash2"
run() { echo "=== $1"; py -u ft-speedprobe.py "$@" 2>&1 | grep -v loading | grep -E "Free memory after|Allocating|MoE slot pools|run \.\.\.|ERROR|rror:" | cut -c1-260; }
run h-mtp3-c4300 --no-base-args --args "$FS $CARD --moe-cache-size 4300" --tests decode,p5k
run h-mtp3-cauto --no-base-args --args "$FS $CARD" --tests decode,p5k
run h-df3-cauto  --no-base-args --args "$FS $DF" --tests decode,p5k
echo "=== done"
