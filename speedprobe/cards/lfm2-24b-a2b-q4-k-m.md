# Model card: LFM2-24B-A2B-Q4_K_M.gguf

Fitted 2026-11-13 14:50 by `scripts/ft-fit.py` on RX 9070 16 GB / 31.9 GiB RAM, every run at context 65536 (--max-seq-len-override).

## Recommended launch

```
ft serve --model-path "E:\ai-models\ft-models\LiquidAI_LFM2-24B-A2B-GGUF\LFM2-24B-A2B-Q4_K_M.gguf" --moe-cpu-layers auto --moe-strategy hybrid --memory-ratio 0.80 --moe-cache-auto --kv-reserve-tokens 65536
```

- Decode: **85.07 tok/s** as stamped (this exact line in one run); ladder run of the same config 123.8 tok/s (plain decode); plain best 102.99 tok/s
- As-stamped run (decode + 5k prompt): decode 85.07 tok/s, p5k 71.84 tok/s; minRAM 2.94 GiB, maxVRAM 15.37 GiB
- Context floor 65536 tokens: **met** (65536 tokens fit)

## Model

- lfm2moe, 40 layers, 64 experts (4 used), file 13.42 GiB (experts 12.89, dense 0.53)
- Expert tensors: {'Q6_K': 18, 'Q4_K': 96}
- KV ~20 KiB/token (bf16 estimate): 32k context ~0.625 GiB
- Vision: none
- MTP: none on disk
- Size-check verdict: hybrid (host need 17.89 GiB text, 17.89 with vision)

## Runs

| stage | config | result |
|---|---|---|
| 4-ratio | `--moe-strategy hybrid --memory-ratio 0.80` (decode) | decode 102.99 tok/s; minRAM 3.7 GiB, maxVRAM 15.38 GiB |
| 4-ratio | `--moe-strategy hybrid --memory-ratio 0.85` (decode) | decode 81.85 tok/s; minRAM 2.66 GiB, maxVRAM 15.46 GiB |
| 6-slots | `--moe-cpu-layers auto --moe-strategy hybrid --memory-ratio 0.80 --moe-cache-auto --kv-reserve-tokens 65536` (decode,p5k) | decode 123.8 tok/s, p5k 83.05 tok/s; minRAM 2.92 GiB, maxVRAM 15.37 GiB |
| 7-final | `--moe-strategy hybrid --memory-ratio 0.80 --moe-cache-auto --kv-reserve-tokens 65536` (decode,p5k) | decode 85.07 tok/s, p5k 71.84 tok/s; minRAM 2.94 GiB, maxVRAM 15.37 GiB |
| 8-ctx | `--moe-strategy hybrid --memory-ratio 0.80 --moe-cache-auto --kv-reserve-tokens 65536` (decode,p64k) | decode 113.29 tok/s, p64k 45.38 tok/s; minRAM 2.43 GiB, maxVRAM 15.49 GiB |
| 8-kv8 | `--moe-strategy hybrid --memory-ratio 0.80 --moe-cache-auto --kv-reserve-tokens 65536 --kv-cache-dtype fp8_per_token_head` (decode) | decode 125.04 tok/s; minRAM 3.68 GiB, maxVRAM 15.38 GiB |

## Context

- Confirmed context on top of the winner: **65536 tokens** (engine KV 65539 tokens; model max 128000); fill test 64K prompt, 45.38 tok/s decode at that depth

## 8-bit KV (lossy, not in the launch line)

- `--kv-cache-dtype fp8_per_token_head` on the as-stamped config: KV **65813 tokens** (16-bit: 65539), decode 125.04 tok/s (16-bit: 85.07), text DIFFERS on the decode test.
- It changes what the model sees of its own context, so adding it is your call; the decode test is too short to judge quality.

## Expert slots against context

--slots-mode auto: the engine sized the slots itself (--moe-cache-auto, KV floor 65536 tokens): **2036 slots**. Loaded at context 128000.

| slots | decode tok/s | 5K prefill tok/s | KV tokens | max VRAM GiB |
|---|---|---|---|---|
| 2036 | 123.8 | 1954.1 | 65539 | 15.37 |

## Notes

- no MTP draft head on disk; spec decoding skipped
- no tune result for class lfm2moe-moe: no flags applied. Run once with --tune (BENCH_FT_FIT_TUNE=1).
