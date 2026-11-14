# Model card: Groot-19B-A3B-261023-GGUF-Q4_K_M.gguf

Fitted 2026-11-12 18:15 by `scripts/ft-fit.py` on RX 9070 16 GB / 31.9 GiB RAM, every run at context 65536 (--max-seq-len-override).

## Recommended launch

```
ft serve --model-path "E:\ai-models\ft-models\GrootTeam_Groot-19B-A3B-261023-GGUF\Groot-19B-A3B-261023-GGUF-Q4_K_M.gguf" --moe-strategy fused --memory-ratio 0.90 --mm-disable vision
```

- Decode: **67.48 tok/s** (plain decode); plain best 67.48 tok/s
- Fit check (decode + 5k prompt): FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-confirm-novis.log; minRAM 18.01 GiB, maxVRAM 1.5 GiB

## Model

- qwen35moe, 41 layers, 128 experts (8 used), file 10.85 GiB (experts 9.33, dense 1.52)
- Expert tensors: {'Q6_K': 21, 'Q4_K': 102}
- KV ~20 KiB/token (bf16 estimate): 32k context ~0.625 GiB
- Vision: mmproj-Groot-19B-A3B-261023-GGUF-Q4_K_M.gguf (0.84 GiB)
- MTP: built-in
- Size-check verdict: fused (host need 14.33 GiB text, 15.17 with vision)

## Runs

| stage | config | result |
|---|---|---|
| 4-ratio | `--moe-strategy fused --memory-ratio 0.90` (decode) | decode 67.48 tok/s; minRAM 5.4 GiB, maxVRAM 15.5 GiB |
| 5-spec | `--moe-strategy fused --memory-ratio 0.90 --speculative-algorithm mtp --speculative-num-steps 2` (decode) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-mtp2.log; minRAM 18.03 GiB, maxVRAM 1.5 GiB |
| 6-flag | `--moe-strategy fused --memory-ratio 0.90 --moe-cpu-threads 6` (decode) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-cputhreads6.log; minRAM 18.24 GiB, maxVRAM 1.49 GiB |
| 6-flag | `--moe-strategy fused --memory-ratio 0.90 --moe-cpu-threads 8` (decode) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-cputhreads8.log; minRAM 18.28 GiB, maxVRAM 1.5 GiB |
| 6-flag | `--moe-strategy fused --memory-ratio 0.90 --moe-hybrid-max-fetch 0` (decode) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-hybridmaxfetch-0.log; minRAM 18.08 GiB, maxVRAM 1.5 GiB |
| 6-flag | `--moe-strategy fused --memory-ratio 0.90 --moe-hybrid-max-fetch 2` (decode) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-hybridmaxfetch-2.log; minRAM 18.42 GiB, maxVRAM 1.48 GiB |
| 6-flag | `--moe-strategy fused --memory-ratio 0.90 --moe-hybrid-max-fetch 4` (decode) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-hybridmaxfetch-4.log; minRAM 18.19 GiB, maxVRAM 1.49 GiB |
| 6-flag | `--moe-strategy fused --memory-ratio 0.90 --moe-locked-max-fetch 4` (decode) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-lockedmaxfetch-4.log; minRAM 17.93 GiB, maxVRAM 1.5 GiB |
| 6-flag | `--moe-strategy fused --memory-ratio 0.90 --moe-locked-max-fetch 16` (decode) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-lockedmaxfetch-16.log; minRAM 18.1 GiB, maxVRAM 1.49 GiB |
| 6-flag | `--moe-strategy fused --memory-ratio 0.90` (decode,p5k) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-ref-decode-p5k.log; minRAM 18.01 GiB, maxVRAM 1.5 GiB |
| 6p-parallel | `--moe-strategy fused --memory-ratio 0.90 --max-running-requests 2 --cuda-graph-max-bs 2` {'FT_GRAPH_MAX_BS': '2'} (decode,c2) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-par2.log; minRAM 17.98 GiB, maxVRAM 1.5 GiB |
| 7-confirm | `--moe-strategy fused --memory-ratio 0.80` +vision (decode,p5k) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-confirm0-vis.log; minRAM 17.95 GiB, maxVRAM 1.49 GiB |
| 7-confirm | `--moe-strategy fused --memory-ratio 0.80` +vision (decode,p5k) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-confirm1-vis.log; minRAM 17.91 GiB, maxVRAM 1.5 GiB |
| 7-confirm | `--moe-strategy fused --memory-ratio 0.90` (decode,p5k) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-confirm-novis.log; minRAM 18.01 GiB, maxVRAM 1.5 GiB |
| 8-ctx | `--moe-strategy fused --memory-ratio 0.90` (decode) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-ctx256k.log; minRAM 17.91 GiB, maxVRAM 1.48 GiB |
| 8-ctx | `--moe-strategy fused --memory-ratio 0.90` (decode) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-ctx192k.log; minRAM 17.91 GiB, maxVRAM 1.49 GiB |
| 8-ctx | `--moe-strategy fused --memory-ratio 0.90` (decode) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-ctx144k.log; minRAM 17.94 GiB, maxVRAM 1.56 GiB |

## Flags

- `cpu-threads-6` (safe, lossless True): FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-cputhreads6.log; minRAM 18.24 GiB, maxVRAM 1.49 GiB
- `cpu-threads-8` (safe, lossless True): FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-cputhreads8.log; minRAM 18.28 GiB, maxVRAM 1.5 GiB
- `fused-combine` (candidate, lossless unknown): not run (candidate)
- `hybrid-max-fetch` (safe, lossless True): `--moe-hybrid-max-fetch 0`: FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-hybridmaxfetch-0.log; minRAM 18.08 GiB, maxVRAM 1.5 GiB | `--moe-hybrid-max-fetch 2`: FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-hybridmaxfetch-2.log; minRAM 18.42 GiB, maxVRAM 1.48 GiB | `--moe-hybrid-max-fetch 4`: FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-hybridmaxfetch-4.log; minRAM 18.19 GiB, maxVRAM 1.49 GiB
- `locked-max-fetch` (safe, lossless True): `--moe-locked-max-fetch 4`: FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-lockedmaxfetch-4.log; minRAM 17.93 GiB, maxVRAM 1.5 GiB | `--moe-locked-max-fetch 16`: FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-lockedmaxfetch-16.log; minRAM 18.1 GiB, maxVRAM 1.49 GiB
- `prefill-chunk` (safe, lossless True): not run (the plain reference with tests decode,p5k failed: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-ref-decode-p5k.log)
- `cache-scope-per-layer` (safe, lossless True): not run (the plain reference with tests decode,p5k failed: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-ref-decode-p5k.log)
- `prefill-hit-d2d` (candidate, lossless True): not run (candidate)
- `prefill-no-overlap` (candidate, lossless True): not run (candidate)
- `cache-split-profile` (candidate, lossless True): not run (candidate)

## Parallel requests

One request decodes at 67.48 tok/s. Best: **1 at once** (more requests gave no more total speed). Not in the launch line: the context per request is your call. To use N: `--max-running-requests N --cuda-graph-max-bs N` and FT_GRAPH_MAX_BS=N.

| requests | total tok/s | per request tok/s | one request tok/s | KV tokens | max VRAM GiB | min RAM GiB |
|---|---|---|---|---|---|---|
| 2 | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km |  |  |  | 1.5 | 17.98 |

## Notes

- MTP k=2 failed: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-mtp2.log
- parallel stage: 2 requests at once failed: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-par2.log
- confirm with vision at --moe-strategy fused --memory-ratio 0.90 failed: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-confirm0-vis.log
- confirm with vision at --moe-strategy fused --memory-ratio 0.85 failed: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-confirm1-vis.log
- vision does not fit with this config: launch with --mm-disable vision
- context 262144 failed to load: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-ctx256k.log
- context 196608 failed to load: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-ctx192k.log
- context 147456 failed to load: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-groot19-q4km-fused-r90-ctx144k.log
