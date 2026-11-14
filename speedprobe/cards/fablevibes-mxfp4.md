# Model card: fablevibes_mxfp4.gguf

Fitted 2026-11-13 06:57 by `scripts/ft-fit.py` on RX 9070 16 GB / 31.9 GiB RAM, every run at context 65536 (--max-seq-len-override).

## Recommended launch

```
ft serve --model-path "E:\ai-hot\fablevibes_mxfp4.gguf" --moe-strategy fused --memory-ratio 0.90
```

- Decode: **73.6 tok/s** as stamped (this exact line in one run); ladder run of the same config 72.56 tok/s (plain decode); plain best 72.56 tok/s
- As-stamped run (decode + 5k prompt): decode 73.6 tok/s, p5k 71.79 tok/s; minRAM 6.7 GiB, maxVRAM 15.48 GiB
- Context floor 65536 tokens: **met** (204800 tokens fit)

## Model

- qwen35moe, 40 layers, 90 experts (8 used), file 8.04 GiB (experts 5.6, dense 2.43)
- Expert tensors: {'MXFP4': 120}
- KV ~20 KiB/token (bf16 estimate): 32k context ~0.625 GiB
- Vision: none
- MTP: mtp-Qwen3.6-35B-A3B-Q4_0.gguf (base model head, other expert count)
- Size-check verdict: fused (host need 10.6 GiB text, 10.6 with vision)

## Runs

| stage | config | result |
|---|---|---|
| 4-ratio | `--moe-strategy fused --memory-ratio 0.90` (decode) | decode 72.56 tok/s; minRAM 7.21 GiB, maxVRAM 15.44 GiB |
| 5-spec | `--moe-strategy fused --memory-ratio 0.90 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-draft-model-path T:\ai-models\gguf\local\mtp-Qwen3.6-35B-A3B-Q4_0.gguf` (decode) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-fablevibes-mxfp4-fused-r90-mtp2.log; minRAM 9.08 GiB, maxVRAM 10.48 GiB |
| 7-final | `--moe-strategy fused --memory-ratio 0.90` (decode,p5k) | decode 73.6 tok/s, p5k 71.79 tok/s; minRAM 6.7 GiB, maxVRAM 15.48 GiB |
| 8-ctx | `--moe-strategy fused --memory-ratio 0.90` (decode,p239k) | FAIL: timed out; minRAM 5.82 GiB, maxVRAM 15.46 GiB |
| 8-ctx | `--moe-strategy fused --memory-ratio 0.90` (decode,p203k) | decode 66.46 tok/s, p203k 45.6 tok/s; minRAM 5.46 GiB, maxVRAM 15.45 GiB |
| 8-kv8 | `--moe-strategy fused --memory-ratio 0.90 --kv-cache-dtype fp8_per_token_head` (decode) | decode 77.02 tok/s; minRAM 6.58 GiB, maxVRAM 15.44 GiB |

## Context

- Confirmed context on top of the winner: **204800 tokens** (engine KV 241391 tokens; model max 262144); fill test 203K prompt, 45.6 tok/s decode at that depth

## 8-bit KV (lossy, not in the launch line)

- `--kv-cache-dtype fp8_per_token_head` on the as-stamped config: KV **475355 tokens** (16-bit: 241391), decode 77.02 tok/s (16-bit: 73.6), text DIFFERS on the decode test.
- It changes what the model sees of its own context, so adding it is your call; the decode test is too short to judge quality.

## Notes

- MTP k=2 failed: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-fablevibes-mxfp4-fused-r90-mtp2.log
- no tune result for class qwen35moe-moe: no flags applied. Run once with --tune (BENCH_FT_FIT_TUNE=1).
- 239K prompt on a 241391-token context failed: timed out
