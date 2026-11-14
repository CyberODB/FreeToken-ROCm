# Model card: Tiel-Coder-35B-A3B-MTP-UD-Q4_K_S.gguf

Fitted 2026-11-09 00:21 by `scripts/ft-fit.py` on RX 9070 16 GB / 31.9 GiB RAM, every run at context 65536 (--max-seq-len-override).

## Recommended launch

```
ft serve --model-path "E:\ai-models\ft-models\peculiar-ragdoll_Tiel-Coder-35B-A3B-GGUF-MTP\Tiel-Coder-35B-A3B-MTP-UD-Q4_K_S.gguf" --moe-cpu-layers auto --moe-cache-size 5000 --moe-strategy hybrid --memory-ratio 0.80 --mm-disable vision
```

- Decode: **84.91 tok/s** (plain decode); plain best 84.91 tok/s
- Fit check (decode + 5k prompt): decode 85.27 tok/s, p5k 81.98 tok/s; minRAM 3.2 GiB, maxVRAM 15.5 GiB

## Model

- qwen35moe, 41 layers, 256 experts (8 used), file 19.81 GiB (experts 17.39, dense 2.42)
- Expert tensors: {'Q4_K': 117, 'Q6_K': 3, 'Q3_K': 3}
- KV ~20 KiB/token (bf16 estimate): 32k context ~0.625 GiB
- Vision: mmproj-BF16.gguf (0.84 GiB)
- MTP: built-in
- Size-check verdict: tight (host need 22.39 GiB text, 23.23 with vision)

## Runs

| stage | config | result |
|---|---|---|
| 4-ratio | `--moe-strategy hybrid --memory-ratio 0.80` (decode) | decode 84.91 tok/s; minRAM 2.68 GiB, maxVRAM 14.73 GiB |
| 4-ratio | `--moe-strategy hybrid --memory-ratio 0.85` (decode) | decode 84.52 tok/s; minRAM 2.65 GiB, maxVRAM 15.52 GiB |
| 5-spec | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2` (decode) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-tiel-coder-35b-a3b-mtp-ud-q4-k-s-hybrid-r80-mtp2.log; minRAM 3.4 GiB, maxVRAM 13.03 GiB |
| 6-flag | `--moe-strategy hybrid --memory-ratio 0.80 --moe-cpu-threads 6` (decode) | decode 85.61 tok/s; minRAM 2.71 GiB, maxVRAM 14.73 GiB |
| 6-flag | `--moe-strategy hybrid --memory-ratio 0.80 --moe-cpu-threads 8` (decode) | decode 84.18 tok/s; minRAM 2.73 GiB, maxVRAM 14.73 GiB |
| 7-confirm | `--moe-strategy hybrid --memory-ratio 0.70` +vision (decode,p5k) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-tiel-coder-35b-a3b-mtp-ud-q4-k-s-hybrid-r80-confirm0-vis.log; minRAM 1.62 GiB, maxVRAM 13.1 GiB |
| 7-confirm | `--moe-strategy hybrid --memory-ratio 0.70` +vision (decode,p5k) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-tiel-coder-35b-a3b-mtp-ud-q4-k-s-hybrid-r80-confirm1-vis.log; minRAM 1.53 GiB, maxVRAM 13.1 GiB |
| 7-confirm | `--moe-strategy hybrid --memory-ratio 0.80` (decode,p5k) | decode 85.27 tok/s, p5k 81.98 tok/s; minRAM 3.2 GiB, maxVRAM 15.5 GiB |
| 8-ctx | `--moe-strategy hybrid --memory-ratio 0.80` +vision (decode) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-tiel-coder-35b-a3b-mtp-ud-q4-k-s-hybrid-r80-ctx256k.log; minRAM 1.46 GiB, maxVRAM 13.1 GiB |
| 8-ctx | `--moe-strategy hybrid --memory-ratio 0.80` +vision (decode) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-tiel-coder-35b-a3b-mtp-ud-q4-k-s-hybrid-r80-ctx192k.log; minRAM 1.47 GiB, maxVRAM 13.1 GiB |
| 8-ctx | `--moe-strategy hybrid --memory-ratio 0.80` +vision (decode) | FAIL: available RAM 1.33 GiB under 1.5 -> The guard stopped it: host RAM ran critically low. Try: close big apps, lower the context or KV, or a smaller quant.; minRAM 1.42 GiB, maxVRAM 13.1 GiB |

## Flags

- `cpu-threads-6` (safe, lossless True): decode 85.61 tok/s; minRAM 2.71 GiB, maxVRAM 14.73 GiB; text identical
- `cpu-threads-8` (safe, lossless True): decode 84.18 tok/s; minRAM 2.73 GiB, maxVRAM 14.73 GiB; text identical
- `fused-combine` (candidate, lossless unknown): not run (candidate)
- `hybrid-max-fetch` (candidate, lossless True): not run (candidate)

## Notes

- MTP k=2 failed: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-tiel-coder-35b-a3b-mtp-ud-q4-k-s-hybrid-r80-mtp2.log
- confirm with vision at --moe-strategy hybrid --memory-ratio 0.80 failed: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-tiel-coder-35b-a3b-mtp-ud-q4-k-s-hybrid-r80-confirm0-vis.log
- confirm with vision at --moe-strategy hybrid --memory-ratio 0.75 failed: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-tiel-coder-35b-a3b-mtp-ud-q4-k-s-hybrid-r80-confirm1-vis.log
- vision does not fit with this config: launch with --mm-disable vision
- context 262144 failed to load: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-tiel-coder-35b-a3b-mtp-ud-q4-k-s-hybrid-r80-ctx256k.log
- context 196608 failed to load: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-tiel-coder-35b-a3b-mtp-ud-q4-k-s-hybrid-r80-ctx192k.log
- context 147456 failed to load: available RAM 1.33 GiB under 1.5
