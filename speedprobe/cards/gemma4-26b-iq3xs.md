# Model card: google_gemma-4-26B-A4B-it-GGUF-IQ3_XS.gguf

Fitted 2026-11-08 05:55 by `scripts/ft-fit.py` on RX 9070 16 GB / 31.9 GiB RAM, every run at context 65536 (--max-seq-len-override).

**No config loaded.**

## Model

- gemma4, 30 layers, 128 experts (8 used), file 11.57 GiB (experts 9.98, dense 1.58)
- Expert tensors: {'Q5_0': 14, 'IQ3_S': 7, 'IQ3_XXS': 23, 'IQ4_NL': 16}
- KV ~0 KiB/token (bf16 estimate): 32k context ~0.0 GiB
- Vision: mmproj-google_gemma-4-26B-A4B-it-GGUF-IQ3_XS.gguf (0.75 GiB)
- MTP: none on disk
- Size-check verdict: tight (host need 14.98 GiB text, 15.74 with vision)

## Runs

| stage | config | result |
|---|---|---|
| 4-ratio | `--moe-strategy hybrid --memory-ratio 0.80` (decode) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-gemma4-26b-iq3xs-hybrid-r80.log; minRAM 3.56 GiB, maxVRAM 4.16 GiB |

## Notes

- hybrid at ratio 0.80 failed: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-gemma4-26b-iq3xs-hybrid-r80.log
- no config loaded; see the errors above
