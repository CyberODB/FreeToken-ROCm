# Model card: Qwen3.6-Whittle-25B-A3B-GGUF-IQ4_XS.gguf

Fitted 2026-11-07 08:25 by `scripts/ft-fit.py` on RX 9070 16 GB / 31.9 GiB RAM.

## Recommended launch

```
ft serve --model-path "E:/ai-models/ft-models/mradermacher_Qwen3.6-Whittle-25B-A3B-GGUF/Qwen3.6-Whittle-25B-A3B-GGUF-IQ4_XS.gguf" --moe-cpu-layers auto --moe-cache-size 5000 --moe-strategy hybrid --memory-ratio 0.80
```

- Decode: **92.21 tok/s** (plain decode); plain best 92.21 tok/s
- Fit check (decode + 5k prompt): not run

## Model

- qwen35moe, 40 layers, 180 experts (8 used), file 12.82 GiB (experts 11.34, dense 1.47)
- Expert tensors: {'Q5_K': 5, 'IQ4_XS': 115}
- KV ~20 KiB/token (bf16 estimate): 32k context ~0.625 GiB
- Vision: none
- MTP: mtp-Qwen3.6-35B-A3B-Q4_0.gguf (base model head, other expert count)
- Size-check verdict: fused (host need 16.34 GiB text, 16.34 with vision)

## Runs

| stage | config | result |
|---|---|---|
| 4-ratio | `--moe-strategy fused --memory-ratio 0.80` (decode) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-qwen3-6-whittle-25b-a3b-gguf-iq4-xs-fused-r80.log; minRAM 3.6 GiB, maxVRAM 14.3 GiB |
| 4-ratio | `--moe-strategy hybrid --memory-ratio 0.80` (decode) | decode 92.21 tok/s; minRAM 2.56 GiB, maxVRAM 14.56 GiB |
| 4-ratio | `--moe-strategy hybrid --memory-ratio 0.85` (decode) | decode 91.82 tok/s; minRAM 3.47 GiB, maxVRAM 15.37 GiB |

## Notes

- fused at ratio 0.80 failed: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-qwen3-6-whittle-25b-a3b-gguf-iq4-xs-fused-r80.log
