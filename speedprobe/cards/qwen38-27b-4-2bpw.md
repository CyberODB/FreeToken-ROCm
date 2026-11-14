# Model card: Qwen3.8-27B-GGUF-4.2BPW-16GB-4.2BPW-16GB.gguf

Fitted 2026-11-07 19:07 by `scripts/ft-fit.py` on RX 9070 16 GB / 31.9 GiB RAM, every run at context 65536 (--max-seq-len-override).

**No config loaded.**

## Model

- qwen35, 65 layers, None experts (None used), file 11.87 GiB (experts 0.0, dense 11.87)
- Expert tensors: {}
- KV ~64 KiB/token (bf16 estimate): 32k context ~2.0 GiB
- Vision: none
- MTP: built-in
- Size-check verdict: hybrid (host need 5.0 GiB text, 5.0 with vision)

## Runs

| stage | config | result |
|---|---|---|
| 4-ratio | `--moe-strategy hybrid --memory-ratio 0.80` (decode) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-qwen38-27b-4-2bpw-hybrid-r80.log; minRAM 4.42 GiB, maxVRAM 13.65 GiB |

## Notes

- hybrid at ratio 0.80 failed: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-qwen38-27b-4-2bpw-hybrid-r80.log
- no config loaded; see the errors above
