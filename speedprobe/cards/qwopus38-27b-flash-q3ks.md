# Model card: Qwopus3.8-27B-Flash-V2-GGUF-MTP-Q3_K_S.gguf

Fitted 2026-11-08 22:02 by `scripts/ft-fit.py` on RX 9070 16 GB / 31.9 GiB RAM, every run at context 65536 (--max-seq-len-override).

## Recommended launch

```
ft serve --model-path "E:\ai-models\ft-models\Jackrong_Qwopus3.8-27B-Flash-V2-GGUF\Qwopus3.8-27B-Flash-V2-GGUF-MTP-Q3_K_S.gguf" --moe-cpu-layers auto --moe-cache-size 5000 --memory-ratio 0.90 --max-running-requests 1 --speculative-algorithm mtp --speculative-num-steps 2
```

- Decode: **37.36 tok/s** (MTP, not byte-identical to plain decode); plain best 30.31 tok/s
- Fit check (decode + 5k prompt): decode 28.47 tok/s, p5k 28.08 tok/s; minRAM 3.62 GiB, maxVRAM 15.6 GiB

## Model

- qwen35, 65 layers, None experts (None used), file 11.4 GiB (experts 0.0, dense 11.4)
- Expert tensors: {}
- KV ~64 KiB/token (bf16 estimate): 32k context ~2.0 GiB
- Vision: none
- MTP: built-in
- Size-check verdict: fused (host need 5.0 GiB text, 5.0 with vision)

## Runs

| stage | config | result |
|---|---|---|
| 4-dense | `--memory-ratio 0.90` (decode) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-qwopus38-27b-flash-q3ks-dense-r90.log; minRAM 4.92 GiB, maxVRAM 13.48 GiB |
| 4-dense | `--memory-ratio 0.90 --max-running-requests 1` (decode) | decode 30.31 tok/s; minRAM 2.65 GiB, maxVRAM 15.6 GiB |
| 5-spec | `--memory-ratio 0.90 --max-running-requests 1 --speculative-algorithm mtp --speculative-num-steps 2` (decode) | decode 37.36 tok/s; minRAM 4.04 GiB, maxVRAM 15.64 GiB |
| 5-spec | `--memory-ratio 0.90 --max-running-requests 1 --speculative-algorithm mtp --speculative-num-steps 3` (decode) | decode 35.85 tok/s; minRAM 4.79 GiB, maxVRAM 15.62 GiB |
| 7-confirm | `--memory-ratio 0.90 --max-running-requests 1` (decode,p5k) | decode 28.47 tok/s, p5k 28.08 tok/s; minRAM 3.62 GiB, maxVRAM 15.6 GiB |
| 8-ctx | `--memory-ratio 0.90 --max-running-requests 1 --speculative-algorithm mtp --speculative-num-steps 2` (decode) | decode 38.38 tok/s; minRAM 4.88 GiB, maxVRAM 15.64 GiB |
| 8-ctx | `--memory-ratio 0.90 --max-running-requests 1 --speculative-algorithm mtp --speculative-num-steps 2` (decode,p20k) | decode 36.97 tok/s, p20k None tok/s; minRAM 4.9 GiB, maxVRAM 15.62 GiB |

## Context

- Confirmed context on top of the winner: **21504 tokens** (engine KV 21684 tokens; model max 262144); fill test 20K prompt

## Flags

- `cpu-threads-6` (safe, lossless True): not run (MoE-only flag; dense model)
- `cpu-threads-8` (safe, lossless True): not run (MoE-only flag; dense model)
- `fused-combine` (candidate, lossless unknown): not run (candidate)
- `hybrid-max-fetch` (candidate, lossless True): not run (MoE-only flag; dense model)

## Notes

- dense --memory-ratio 0.90 failed: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-qwopus38-27b-flash-q3ks-dense-r90.log
