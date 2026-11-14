# Model card: LiquidAI_LFM2-24B-A2B-Q5_K_L.gguf

Fitted 2026-11-12 17:40 by `scripts/ft-fit.py` on RX 9070 16 GB / 31.9 GiB RAM, every run at context 65536 (--max-seq-len-override).

**No config loaded.**

## Model

- lfm2moe, 40 layers, 64 experts (4 used), file 15.8 GiB (experts 15.14, dense 0.65)
- Expert tensors: {'Q6_K': 18, 'Q5_K': 96}
- KV ~0 KiB/token (bf16 estimate): 32k context ~0.0 GiB
- Vision: none
- MTP: none on disk
- Size-check verdict: tight (host need 20.14 GiB text, 20.14 with vision)

## Runs

| stage | config | result |
|---|---|---|
| 4-ratio | `--moe-strategy hybrid --memory-ratio 0.80` (decode) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-liquidai-lfm2-24b-a2b-q5-k-l-hybrid-r80.log; minRAM 2.9 GiB, maxVRAM 2.94 GiB |

## Notes

- hybrid at ratio 0.80 failed: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-liquidai-lfm2-24b-a2b-q5-k-l-hybrid-r80.log
- no config loaded; see the errors above
