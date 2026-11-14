# Model card: LFM2.5-24B-A2B-NVFP4-GGUF-lfm2.5-24b-a2b-nvfp4.gguf

Fitted 2026-11-13 11:08 by `scripts/ft-fit.py` on RX 9070 16 GB / 31.9 GiB RAM, every run at context 65536 (--max-seq-len-override).

**No config loaded.**

## Model

- lfm2moe, 40 layers, 64 experts (4 used), file 12.64 GiB (experts 12.02, dense 0.61)
- Expert tensors: {'NVFP4': 114}
- KV ~20 KiB/token (bf16 estimate): 32k context ~0.625 GiB
- Vision: none
- MTP: none on disk
- Size-check verdict: tight (host need 17.02 GiB text, 17.02 with vision)

## Runs

| stage | config | result |
|---|---|---|
| 4-ratio | `--moe-strategy hybrid --memory-ratio 0.80` (decode) | FAIL: available RAM 0.86 GiB under 1.1 -> The guard stopped it: host RAM ran critically low. Try: close big apps, lower the context or KV, or a smaller quant.; minRAM 0.86 GiB, maxVRAM 15.41 GiB |

## Notes

- hybrid at ratio 0.80 failed: available RAM 0.86 GiB under 1.1 -> The guard stopped it: host RAM ran critically low. Try: close big apps, lower the context or KV, or a smaller quant.
- no config loaded; see the errors above
