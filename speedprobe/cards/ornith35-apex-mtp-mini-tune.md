# Tune result: class qwen35moe-moe (run on ornith35_apex-mtp-mini.gguf)

Fitted 2026-11-14 06:05 by `scripts/ft-fit.py` on RX 9070 16 GB / 31.9 GiB RAM, every run at context 65536 (--max-seq-len-override).

**Not a config to stamp.** Flags and the parallel report for every model of this class, measured on the plain config `--moe-strategy hybrid --memory-ratio 0.80` (111.28 tok/s). A fit of such a model applies the kept flags in its as-stamped run.

## Model

- qwen35moe, 41 layers, 256 experts (8 used), file 13.37 GiB (experts 12.05, dense 1.32)
- Expert tensors: {'Q3_K': 60, 'IQ2_S': 60, 'Q8_0': 3}
- KV ~20 KiB/token (bf16 estimate): 32k context ~0.625 GiB
- Vision: none
- MTP: built-in
- Size-check verdict: tight (host need 17.05 GiB text, 17.05 with vision)

## Runs

| stage | config | result |
|---|---|---|
| 4-ratio | `--moe-strategy hybrid --memory-ratio 0.80` (decode) | decode 111.28 tok/s; minRAM 2.42 GiB, maxVRAM 14.64 GiB |
| 4-ratio | `--moe-strategy hybrid --memory-ratio 0.85` (decode) | decode 107.24 tok/s; minRAM 2.85 GiB, maxVRAM 15.46 GiB |
| 6-flag | `--moe-strategy hybrid --memory-ratio 0.80 --moe-cpu-threads 6` (decode) | decode 111.42 tok/s; minRAM 2.89 GiB, maxVRAM 14.64 GiB |
| 6-flag | `--moe-strategy hybrid --memory-ratio 0.80 --moe-cpu-threads 8` (decode) | decode 108.56 tok/s; minRAM 2.78 GiB, maxVRAM 14.63 GiB |
| 6-flag | `--moe-strategy hybrid --memory-ratio 0.80 --moe-hybrid-max-fetch 0` (decode) | decode 40.33 tok/s; minRAM 2.97 GiB, maxVRAM 14.63 GiB |
| 6-flag | `--moe-strategy hybrid --memory-ratio 0.80 --moe-hybrid-max-fetch 2` (decode) | decode 102.96 tok/s; minRAM 2.84 GiB, maxVRAM 14.64 GiB |
| 6-flag | `--moe-strategy hybrid --memory-ratio 0.80 --moe-hybrid-max-fetch 4` (decode) | decode 105.51 tok/s; minRAM 2.74 GiB, maxVRAM 14.64 GiB |
| 6-flag | `--moe-strategy hybrid --memory-ratio 0.80 --moe-locked-max-fetch 4` (decode) | decode 101.23 tok/s; minRAM 2.72 GiB, maxVRAM 14.66 GiB |
| 6-flag | `--moe-strategy hybrid --memory-ratio 0.80 --moe-locked-max-fetch 16` (decode) | decode 103.21 tok/s; minRAM 2.62 GiB, maxVRAM 14.65 GiB |
| 6-flag | `--moe-strategy hybrid --memory-ratio 0.80` (decode,p5k) | decode 101.17 tok/s, p5k 101.08 tok/s; minRAM 2.55 GiB, maxVRAM 15.24 GiB |
| 6-flag | `--moe-strategy hybrid --memory-ratio 0.80 --max-prefill-length 2048` (decode,p5k) | decode 100.94 tok/s, p5k 93.02 tok/s; minRAM 2.52 GiB, maxVRAM 14.88 GiB |
| 6-flag | `--moe-strategy hybrid --memory-ratio 0.80 --max-prefill-length 4096` (decode,p5k) | decode 106.6 tok/s, p5k 95.98 tok/s; minRAM 2.46 GiB, maxVRAM 15.12 GiB |
| 6-flag | `--moe-strategy hybrid --memory-ratio 0.80 --moe-cache-scope per-layer` (decode,p5k) | decode 97.93 tok/s, p5k 92.51 tok/s; minRAM 2.3 GiB, maxVRAM 15.23 GiB |
| 6p-parallel | `--moe-strategy hybrid --memory-ratio 0.80 --max-running-requests 2 --cuda-graph-max-bs 2` {'FT_GRAPH_MAX_BS': '2'} (decode,c2) | decode 102.03 tok/s, c2 160.53 tok/s; minRAM 1.4 GiB, maxVRAM 14.65 GiB |
| 6p-parallel | `--moe-strategy hybrid --memory-ratio 0.80 --max-running-requests 4 --cuda-graph-max-bs 4` {'FT_GRAPH_MAX_BS': '4'} (decode,c4) | decode 110.75 tok/s, c4 245.7 tok/s; minRAM 1.9 GiB, maxVRAM 14.68 GiB |
| 6p-parallel | `--moe-strategy hybrid --memory-ratio 0.80 --max-running-requests 8 --cuda-graph-max-bs 8` {'FT_GRAPH_MAX_BS': '8'} (decode,c8) | decode 115.5 tok/s, c8 267.0 tok/s; minRAM 1.45 GiB, maxVRAM 14.62 GiB |
| 6p-parallel | `--moe-strategy hybrid --memory-ratio 0.80 --max-running-requests 4 --cuda-graph-max-bs 1` {'FT_GRAPH_MAX_BS': '1'} (decode,c4) | decode 114.74 tok/s, c4 58.99 tok/s; minRAM 1.56 GiB, maxVRAM 14.54 GiB |
| 6p-parallel | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm dflash --speculative-num-steps 3 --speculative-draft-model-path E:/ai-models/ft-models/jzinno_Ornith-1.5-35B-A3B-DFlash2 --max-running-requests 2 --cuda-graph-max-bs 2` {'FT_GRAPH_MAX_BS': '2'} (decode,c2) | decode 98.07 tok/s, c2 110.88 tok/s; minRAM 1.16 GiB, maxVRAM 14.61 GiB |
| 6p-parallel | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm dflash --speculative-num-steps 3 --speculative-draft-model-path E:/ai-models/ft-models/jzinno_Ornith-1.5-35B-A3B-DFlash2 --max-running-requests 4 --cuda-graph-max-bs 4` {'FT_GRAPH_MAX_BS': '4'} (decode,c4) | decode 96.78 tok/s, c4 182.64 tok/s; minRAM 1.18 GiB, maxVRAM 14.67 GiB |

## Flags

- `cpu-threads-6` (safe, lossless True): decode 111.42 tok/s; minRAM 2.89 GiB, maxVRAM 14.64 GiB; text identical
- `cpu-threads-8` (safe, lossless True): decode 108.56 tok/s; minRAM 2.78 GiB, maxVRAM 14.63 GiB; text identical
- `fused-combine` (candidate, lossless unknown): not run (candidate)
- `hybrid-max-fetch` (safe, lossless True): `--moe-hybrid-max-fetch 0`: decode 40.33 tok/s; minRAM 2.97 GiB, maxVRAM 14.63 GiB; TEXT DIFFERS; worse: decode 40.33 vs 111.28 | `--moe-hybrid-max-fetch 2`: decode 102.96 tok/s; minRAM 2.84 GiB, maxVRAM 14.64 GiB; TEXT DIFFERS; worse: decode 102.96 vs 111.28 | `--moe-hybrid-max-fetch 4`: decode 105.51 tok/s; minRAM 2.74 GiB, maxVRAM 14.64 GiB; TEXT DIFFERS; worse: decode 105.51 vs 111.28
- `locked-max-fetch` (safe, lossless True): `--moe-locked-max-fetch 4`: decode 101.23 tok/s; minRAM 2.72 GiB, maxVRAM 14.66 GiB; text identical; worse: decode 101.23 vs 111.28 | `--moe-locked-max-fetch 16`: decode 103.21 tok/s; minRAM 2.62 GiB, maxVRAM 14.65 GiB; text identical; worse: decode 103.21 vs 111.28
- `prefill-chunk` (safe, lossless True): `--max-prefill-length 2048`: decode 100.94 tok/s, p5k 93.02 tok/s; minRAM 2.52 GiB, maxVRAM 14.88 GiB; prefill:p5k 1765.4 vs 2684.4; text identical; worse: 5K prefill 1765.4 vs 2684.4 | `--max-prefill-length 4096`: decode 106.6 tok/s, p5k 95.98 tok/s; minRAM 2.46 GiB, maxVRAM 15.12 GiB; prefill:p5k 2126.7 vs 2684.4; text identical; worse: 5K prefill 2126.7 vs 2684.4
- `cache-scope-per-layer` (safe, lossless True): decode 97.93 tok/s, p5k 92.51 tok/s; minRAM 2.3 GiB, maxVRAM 15.23 GiB; TEXT DIFFERS
- `prefill-hit-d2d` (candidate, lossless True): not run (candidate)
- `prefill-no-overlap` (candidate, lossless True): not run (candidate)
- `cache-split-profile` (candidate, lossless True): not run (candidate)
- `kv-cache-fp8` (candidate, lossless False): not run (candidate)
- `spec-token-map` (candidate, lossless True): not run (candidate)
- `online-quant-df11` (candidate, lossless True): not run (candidate)

## Parallel requests

One request decodes at 111.28 tok/s. Best: **8 at once**. Not in the launch line: the context per request is your call. To use N: `--max-running-requests N --cuda-graph-max-bs N` and FT_GRAPH_MAX_BS=N.

| requests | graph bs | extra args | total tok/s | per request tok/s | one request tok/s | KV tokens | max VRAM GiB | min RAM GiB |
|---|---|---|---|---|---|---|---|---|
| 2 | 2 |  | 160.53 | 80.29 80.11 | 102.03 | 260419 | 14.65 | 1.4 |
| 4 | 4 |  | 245.7 | 61.31 61.34 61.34 61.34 | 110.75 | 222691 | 14.68 | 1.9 |
| 8 | 8 |  | 267.0 | 33.31 33.33 33.33 33.33 33.33 33.33 33.33 33.33 | 115.5 | 147235 | 14.62 | 1.45 |
| 4 | 1 |  | 58.99 | 14.72 14.78 14.78 14.78 | 114.74 | 222691 | 14.54 | 1.56 |
| 2 | 2 | --speculative-algorithm dflash --speculative-num-steps 3 --speculative-draft-model-path E:/ai-models/ft-models/jzinno_Ornith-1.5-35B-A3B-DFlash2 | 110.88 | 58.19 56.63 | 98.07 | 177281 | 14.61 | 1.16 |
| 4 | 4 | --speculative-algorithm dflash --speculative-num-steps 3 --speculative-draft-model-path E:/ai-models/ft-models/jzinno_Ornith-1.5-35B-A3B-DFlash2 | 182.64 | 50.52 50.85 50.17 47.93 | 96.78 | 118491 | 14.67 | 1.18 |

## Notes

- tune run: flags and parallel report saved to E:\freetoken-021\speedprobe\tune\qwen35moe-moe.json; not a config to stamp
