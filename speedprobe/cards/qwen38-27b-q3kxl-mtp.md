# Model card: Qwen3.8-27B-GGUF-UD-Q3_K_XL.gguf

Fitted 2026-11-08 20:32 by `scripts/ft-fit.py` on RX 9070 16 GB / 31.9 GiB RAM, every run at context 65536 (--max-seq-len-override).

**No config loaded.**

## Model

- qwen35, 65 layers, None experts (None used), file 12.23 GiB (experts 0.0, dense 12.23)
- Expert tensors: {}
- KV ~64 KiB/token (bf16 estimate): 32k context ~2.0 GiB
- Vision: mmproj-Qwen3.8-27B-GGUF-UD-Q3_K_XL.gguf (0.86 GiB)
- MTP: built-in
- Size-check verdict: hybrid (host need 5.0 GiB text, 5.86 with vision)

## Runs

| stage | config | result |
|---|---|---|
| 4-ratio | `--moe-strategy hybrid --memory-ratio 0.80` (decode) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-qwen38-27b-q3kxl-mtp-hybrid-r80.log; minRAM 6.08 GiB, maxVRAM 13.97 GiB |

## Notes

- hybrid at ratio 0.80 failed: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-qwen38-27b-q3kxl-mtp-hybrid-r80.log
- no config loaded; see the errors above
