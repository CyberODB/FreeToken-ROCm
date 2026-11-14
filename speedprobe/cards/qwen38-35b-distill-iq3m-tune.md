# Tune result: class qwen35moe-moe (run on qwen38-35b-distill_iq3m.gguf)

Fitted 2026-11-14 06:44 by `scripts/ft-fit.py` on RX 9070 16 GB / 31.9 GiB RAM, every run at context 65536 (--max-seq-len-override).

**Not a config to stamp.** Flags and the parallel report for every model of this class, measured on the plain config `--moe-strategy hybrid --memory-ratio 0.80` (96.59 tok/s). A fit of such a model applies the kept flags in its as-stamped run.

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
| 4-ratio | `--moe-strategy hybrid --memory-ratio 0.80` (decode) | decode 96.59 tok/s; minRAM 3.25 GiB, maxVRAM 15.0 GiB |
| 4-ratio | `--moe-strategy hybrid --memory-ratio 0.85` (decode) | decode 80.68 tok/s; minRAM 3.4 GiB, maxVRAM 15.55 GiB |
| 6-flag | `--moe-strategy hybrid --memory-ratio 0.80 --moe-cpu-threads 6` (decode) | decode 95.28 tok/s; minRAM 4.12 GiB, maxVRAM 14.88 GiB |
| 6-flag | `--moe-strategy hybrid --memory-ratio 0.80 --moe-cpu-threads 8` (decode) | decode 100.37 tok/s; minRAM 4.54 GiB, maxVRAM 14.88 GiB |
| 6-flag | `--moe-strategy hybrid --memory-ratio 0.80 --moe-cpu-threads 8` (decode) | FAIL: available RAM 0.88 GiB under 0.9 -> The guard stopped it: host RAM ran critically low. Try: close big apps, lower the context or KV, or a smaller quant.; minRAM 0.84 GiB, maxVRAM 9.25 GiB |
| 6-flag | `--moe-strategy hybrid --memory-ratio 0.80 --moe-hybrid-max-fetch 0` (decode) | decode 24.58 tok/s; minRAM 4.56 GiB, maxVRAM 14.84 GiB |
| 6-flag | `--moe-strategy hybrid --memory-ratio 0.80 --moe-hybrid-max-fetch 2` (decode) | decode 96.9 tok/s; minRAM 4.27 GiB, maxVRAM 14.84 GiB |
| 6-flag | `--moe-strategy hybrid --memory-ratio 0.80 --moe-hybrid-max-fetch 4` (decode) | decode 94.11 tok/s; minRAM 4.26 GiB, maxVRAM 14.85 GiB |
| 6-flag | `--moe-strategy hybrid --memory-ratio 0.80 --moe-locked-max-fetch 4` (decode) | decode 96.73 tok/s; minRAM 4.34 GiB, maxVRAM 14.84 GiB |
| 6-flag | `--moe-strategy hybrid --memory-ratio 0.80 --moe-locked-max-fetch 16` (decode) | decode 96.38 tok/s; minRAM 4.19 GiB, maxVRAM 14.84 GiB |
| 6-flag | `--moe-strategy hybrid --memory-ratio 0.80` (decode,p5k) | decode 97.44 tok/s, p5k 89.11 tok/s; minRAM 4.12 GiB, maxVRAM 15.48 GiB |
| 6-flag | `--moe-strategy hybrid --memory-ratio 0.80 --max-prefill-length 2048` (decode,p5k) | decode 97.1 tok/s, p5k 89.66 tok/s; minRAM 4.18 GiB, maxVRAM 15.06 GiB |
| 6-flag | `--moe-strategy hybrid --memory-ratio 0.80 --max-prefill-length 4096` (decode,p5k) | FAIL: available RAM 0.79 GiB under 0.9 -> The guard stopped it: host RAM ran critically low. Try: close big apps, lower the context or KV, or a smaller quant.; minRAM 0.72 GiB, maxVRAM 14.34 GiB |
| 6-flag | `--moe-strategy hybrid --memory-ratio 0.80 --moe-cache-scope per-layer` (decode,p5k) | decode 103.11 tok/s, p5k 97.14 tok/s; minRAM 0.85 GiB, maxVRAM 15.11 GiB |
| 6p-parallel | `--moe-strategy hybrid --memory-ratio 0.80 --max-running-requests 2 --cuda-graph-max-bs 2` {'FT_GRAPH_MAX_BS': '2'} (decode,c2) | decode 103.66 tok/s, c2 168.37 tok/s; minRAM 1.17 GiB, maxVRAM 14.5 GiB |
| 6p-parallel | `--moe-strategy hybrid --memory-ratio 0.80 --max-running-requests 4 --cuda-graph-max-bs 4` {'FT_GRAPH_MAX_BS': '4'} (decode,c4) | decode 104.13 tok/s, c4 223.19 tok/s; minRAM 1.45 GiB, maxVRAM 14.55 GiB |
| 6p-parallel | `--moe-strategy hybrid --memory-ratio 0.80 --max-running-requests 8 --cuda-graph-max-bs 8` {'FT_GRAPH_MAX_BS': '8'} (decode,c8) | decode 103.88 tok/s, c8 264.71 tok/s; minRAM 1.72 GiB, maxVRAM 14.58 GiB |
| 6p-parallel | `--moe-strategy hybrid --memory-ratio 0.80 --max-running-requests 4 --cuda-graph-max-bs 1` {'FT_GRAPH_MAX_BS': '1'} (decode,c4) | decode 103.81 tok/s, c4 56.87 tok/s; minRAM 1.81 GiB, maxVRAM 14.51 GiB |

## Flags

- `cpu-threads-6` (safe, lossless True): decode 95.28 tok/s; minRAM 4.12 GiB, maxVRAM 14.88 GiB; text identical
- `cpu-threads-8` (safe, lossless True): decode 100.37 tok/s; minRAM 4.54 GiB, maxVRAM 14.88 GiB; text identical | second run | FAIL: available RAM 0.88 GiB under 0.9 -> The guard stopped it: host RAM ran critically low. Try: close big apps, lower the context or KV, or a smaller quant.; minRAM 0.84 GiB, maxVRAM 9.25 GiB | gain not repeated
- `fused-combine` (candidate, lossless unknown): not run (candidate)
- `hybrid-max-fetch` (safe, lossless True): `--moe-hybrid-max-fetch 0`: decode 24.58 tok/s; minRAM 4.56 GiB, maxVRAM 14.84 GiB; TEXT DIFFERS; worse: decode 24.58 vs 96.59 | `--moe-hybrid-max-fetch 2`: decode 96.9 tok/s; minRAM 4.27 GiB, maxVRAM 14.84 GiB; TEXT DIFFERS | `--moe-hybrid-max-fetch 4`: decode 94.11 tok/s; minRAM 4.26 GiB, maxVRAM 14.85 GiB; TEXT DIFFERS
- `locked-max-fetch` (safe, lossless True): `--moe-locked-max-fetch 4`: decode 96.73 tok/s; minRAM 4.34 GiB, maxVRAM 14.84 GiB; text identical | `--moe-locked-max-fetch 16`: decode 96.38 tok/s; minRAM 4.19 GiB, maxVRAM 14.84 GiB; text identical
- `prefill-chunk` (safe, lossless True): `--max-prefill-length 2048`: decode 97.1 tok/s, p5k 89.66 tok/s; minRAM 4.18 GiB, maxVRAM 15.06 GiB; prefill:p5k 1570.6 vs 2181.7; text identical; worse: 5K prefill 1570.6 vs 2181.7 | `--max-prefill-length 4096`: FAIL: available RAM 0.79 GiB under 0.9 -> The guard stopped it: host RAM ran critically low. Try: close big apps, lower the context or KV, or a smaller quant.; minRAM 0.72 GiB, maxVRAM 14.34 GiB
- `cache-scope-per-layer` (safe, lossless True): decode 103.11 tok/s, p5k 97.14 tok/s; minRAM 0.85 GiB, maxVRAM 15.11 GiB; TEXT DIFFERS
- `prefill-hit-d2d` (candidate, lossless True): not run (candidate)
- `prefill-no-overlap` (candidate, lossless True): not run (candidate)
- `cache-split-profile` (candidate, lossless True): not run (candidate)
- `kv-cache-fp8` (candidate, lossless False): not run (candidate)
- `spec-token-map` (candidate, lossless True): not run (candidate)
- `online-quant-df11` (candidate, lossless True): not run (candidate)

## Parallel requests

One request decodes at 96.59 tok/s. Best: **8 at once**. Not in the launch line: the context per request is your call. To use N: `--max-running-requests N --cuda-graph-max-bs N` and FT_GRAPH_MAX_BS=N.

| requests | graph bs | extra args | total tok/s | per request tok/s | one request tok/s | KV tokens | max VRAM GiB | min RAM GiB |
|---|---|---|---|---|---|---|---|---|
| 2 | 2 |  | 168.37 | 84.02 84.25 | 103.66 | 214339 | 14.5 | 1.17 |
| 4 | 4 |  | 223.19 | 55.69 55.8 55.8 55.8 | 104.13 | 176611 | 14.55 | 1.45 |
| 8 | 8 |  | 264.71 | 33.03 33.07 33.07 33.07 33.07 33.07 33.07 33.07 | 103.88 | 101155 | 14.58 | 1.72 |
| 4 | 1 |  | 56.87 | 14.19 14.26 14.26 14.26 | 103.81 | 176611 | 14.51 | 1.81 |

## Notes

- tune run: flags and parallel report saved to E:\freetoken-021\speedprobe\tune\qwen35moe-moe.json; not a config to stamp
