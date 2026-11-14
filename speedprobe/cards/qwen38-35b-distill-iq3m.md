# Model card: Qwen3.8-35B-A3B-Distill-GGUF-IQ3_M.gguf

Fitted 2026-11-07 20:46 by `scripts/ft-fit.py` on RX 9070 16 GB / 31.9 GiB RAM, every run at context 65536 (--max-seq-len-override).

## Recommended launch

```
ft serve --model-path "E:\ai-models\ft-models\empero-ai_Qwen3.8-35B-A3B-Distill-GGUF\Qwen3.8-35B-A3B-Distill-GGUF-IQ3_M.gguf" --moe-cpu-layers auto --moe-cache-size 5000 --moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2 --moe-cpu-threads 6 --moe-cpu-threads 8
```

- Decode: **114.54 tok/s** (MTP, not byte-identical to plain decode); plain best 92.06 tok/s
- Fit check (decode + 5k prompt): decode 101.72 tok/s, p5k 94.19 tok/s; minRAM 3.6 GiB, maxVRAM 15.58 GiB

## Model

- qwen35moe, 41 layers, 256 experts (8 used), file 15.21 GiB (experts 13.85, dense 1.35)
- Expert tensors: {'Q4_K': 5, 'IQ3_S': 115, 'Q8_0': 3}
- KV ~20 KiB/token (bf16 estimate): 32k context ~0.625 GiB
- Vision: none
- MTP: built-in
- Size-check verdict: tight (host need 18.85 GiB text, 18.85 with vision)

## Runs

| stage | config | result |
|---|---|---|
| 4-ratio | `--moe-strategy hybrid --memory-ratio 0.80` (decode) | decode 92.06 tok/s; minRAM 2.98 GiB, maxVRAM 14.98 GiB |
| 4-ratio | `--moe-strategy hybrid --memory-ratio 0.85` (decode) | decode 85.98 tok/s; minRAM 3.08 GiB, maxVRAM 15.66 GiB |
| 5-spec | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2` (decode) | decode 114.54 tok/s; minRAM 3.65 GiB, maxVRAM 15.08 GiB |
| 5-spec | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 3` (decode) | decode 107.08 tok/s; minRAM 4.26 GiB, maxVRAM 15.06 GiB |
| 6-flag | `--moe-strategy hybrid --memory-ratio 0.80 --moe-cpu-threads 6` (decode) | decode 100.69 tok/s; minRAM 4.26 GiB, maxVRAM 15.01 GiB |
| 6-flag | `--moe-strategy hybrid --memory-ratio 0.80 --moe-cpu-threads 8` (decode) | decode 101.58 tok/s; minRAM 4.34 GiB, maxVRAM 15.01 GiB |
| 7-confirm | `--moe-strategy hybrid --memory-ratio 0.80` (decode,p5k) | decode 101.72 tok/s, p5k 94.19 tok/s; minRAM 3.6 GiB, maxVRAM 15.58 GiB |
| 8-ctx | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2` (decode) | decode 116.95 tok/s; minRAM 4.15 GiB, maxVRAM 15.02 GiB |
| 8-ctx | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2` (decode,p117k) | decode 116.2 tok/s, p117k None tok/s; minRAM 4.38 GiB, maxVRAM 15.02 GiB |

## Context

- Confirmed context on top of the winner: **118784 tokens** (engine KV 119127 tokens; model max 262144); fill test 117K prompt

## Flags

- `cpu-threads-6` (safe, lossless True): decode 100.69 tok/s; minRAM 4.26 GiB, maxVRAM 15.01 GiB; text identical **kept**
- `cpu-threads-8` (safe, lossless True): decode 101.58 tok/s; minRAM 4.34 GiB, maxVRAM 15.01 GiB; text identical **kept**
- `fused-combine` (candidate, lossless unknown): not run (candidate)
- `hybrid-max-fetch` (candidate, lossless True): not run (candidate)
