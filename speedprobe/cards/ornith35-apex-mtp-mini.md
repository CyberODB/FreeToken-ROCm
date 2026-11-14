# Model card: ornith35-apex-mtp-mini.gguf

Fitted 2026-11-13 15:55 by `scripts/ft-fit.py` on RX 9070 16 GB / 31.9 GiB RAM, engine default context (no --ctx).

## Recommended launch

```
ft serve --model-path "E:/ai-models/ollama-view/library_ornith35/ornith35-apex-mtp-mini.gguf" --moe-cpu-layers auto --moe-cache-size 5000 --moe-strategy hybrid --memory-ratio 0.75 --speculative-algorithm dflash --speculative-num-steps 3 --speculative-draft-model-path E:/ai-models/ft-models/jzinno_Ornith-1.5-35B-A3B-DFlash2
```

- Decode: **95.63 tok/s** as stamped (this exact line in one run); ladder run of the same config 81.26 tok/s (DFlash, not byte-identical to plain decode); plain best 75.43 tok/s
- As-stamped run (decode + 5k prompt, vision on): decode 95.63 tok/s, p5k 88.63 tok/s; minRAM 1.21 GiB, maxVRAM 15.42 GiB
- Context floor 65536 tokens: **NOT met** (30720 tokens fit)

## Model

- qwen35moe, 41 layers, 256 experts (8 used), file 13.37 GiB (experts 12.05, dense 1.32)
- Expert tensors: {'Q3_K': 60, 'IQ2_S': 60, 'Q8_0': 3}
- KV ~20 KiB/token (bf16 estimate): 32k context ~0.625 GiB
- Vision: mmproj-ornith35-apex-mtp-mini.gguf (0.84 GiB)
- MTP: built-in
- Size-check verdict: tight-vision (host need 17.05 GiB text, 17.88 with vision)

## Runs

| stage | config | result |
|---|---|---|
| 4-ratio | `--moe-strategy hybrid --memory-ratio 0.80` (decode) | decode 75.43 tok/s; minRAM 4.01 GiB, maxVRAM 15.26 GiB |
| 4-ratio | `--moe-strategy hybrid --memory-ratio 0.85` (decode) | decode 61.67 tok/s; minRAM 3.19 GiB, maxVRAM 15.28 GiB |
| 5-spec | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2` (decode) | decode 73.81 tok/s; minRAM 3.28 GiB, maxVRAM 15.3 GiB |
| 5-spec | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 3` (decode) | decode 62.69 tok/s; minRAM 3.23 GiB, maxVRAM 15.27 GiB |
| 5-spec | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\ornith35-apex-mtp-mini.bin` (decode) | decode 76.88 tok/s; minRAM 3.57 GiB, maxVRAM 15.32 GiB |
| 5-spec | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm dflash --speculative-num-steps 3 --speculative-draft-model-path E:/ai-models/ft-models/jzinno_Ornith-1.5-35B-A3B-DFlash2` (decode) | decode 81.26 tok/s; minRAM 3.52 GiB, maxVRAM 15.31 GiB |
| 5-spec | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm dflash --speculative-num-steps 5 --speculative-draft-model-path E:/ai-models/ft-models/jzinno_Ornith-1.5-35B-A3B-DFlash2` (decode) | decode 62.98 tok/s; minRAM 4.09 GiB, maxVRAM 15.32 GiB |
| 5-spec | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm dflash --speculative-num-steps 7 --speculative-draft-model-path E:/ai-models/ft-models/jzinno_Ornith-1.5-35B-A3B-DFlash2` (decode) | decode 61.86 tok/s; minRAM 4.06 GiB, maxVRAM 15.3 GiB |
| 5-spec | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm dflash --speculative-num-steps 11 --speculative-draft-model-path E:/ai-models/ft-models/jzinno_Ornith-1.5-35B-A3B-DFlash2` (decode) | decode 47.02 tok/s; minRAM 3.82 GiB, maxVRAM 15.45 GiB |
| 5-spec | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm dflash --speculative-num-steps 15 --speculative-draft-model-path E:/ai-models/ft-models/jzinno_Ornith-1.5-35B-A3B-DFlash2` (decode) | decode 44.97 tok/s; minRAM 3.92 GiB, maxVRAM 15.44 GiB |
| 6-slots | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm dflash --speculative-num-steps 3 --speculative-draft-model-path E:/ai-models/ft-models/jzinno_Ornith-1.5-35B-A3B-DFlash2` (decode,p5k) | decode 77.89 tok/s, p5k 51.77 tok/s; minRAM 2.87 GiB, maxVRAM 15.4 GiB |
| 6-slots | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm dflash --speculative-num-steps 3 --speculative-draft-model-path E:/ai-models/ft-models/jzinno_Ornith-1.5-35B-A3B-DFlash2 --moe-cache-size 6500` (decode,p5k) | decode 85.4 tok/s, p5k 76.05 tok/s; minRAM 3.05 GiB, maxVRAM 15.43 GiB |
| 7-final | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm dflash --speculative-num-steps 3 --speculative-draft-model-path E:/ai-models/ft-models/jzinno_Ornith-1.5-35B-A3B-DFlash2` +vision (decode,p5k) | FAIL: available RAM 1.04 GiB under 1.1 -> The guard stopped it: host RAM ran critically low. Try: close big apps, lower the context or KV, or a smaller quant.; minRAM 1.04 GiB, maxVRAM 15.49 GiB |
| 7-final | `--moe-strategy hybrid --memory-ratio 0.75 --speculative-algorithm dflash --speculative-num-steps 3 --speculative-draft-model-path E:/ai-models/ft-models/jzinno_Ornith-1.5-35B-A3B-DFlash2` +vision (decode,p5k) | decode 95.63 tok/s, p5k 88.63 tok/s; minRAM 1.21 GiB, maxVRAM 15.42 GiB |
| 8-ctx | `--moe-strategy hybrid --memory-ratio 0.75 --speculative-algorithm dflash --speculative-num-steps 3 --speculative-draft-model-path E:/ai-models/ft-models/jzinno_Ornith-1.5-35B-A3B-DFlash2` +vision (decode,p51k) | FAIL: available RAM 0.86 GiB under 1.1 -> The guard stopped it: host RAM ran critically low. Try: close big apps, lower the context or KV, or a smaller quant.; minRAM 0.86 GiB, maxVRAM 15.41 GiB |
| 8-ctx | `--moe-strategy hybrid --memory-ratio 0.75 --speculative-algorithm dflash --speculative-num-steps 3 --speculative-draft-model-path E:/ai-models/ft-models/jzinno_Ornith-1.5-35B-A3B-DFlash2` +vision (decode,p42k) | FAIL: available RAM 0.81 GiB under 1.1 -> The guard stopped it: host RAM ran critically low. Try: close big apps, lower the context or KV, or a smaller quant.; minRAM 0.81 GiB, maxVRAM 15.38 GiB |
| 8-ctx | `--moe-strategy hybrid --memory-ratio 0.75 --speculative-algorithm dflash --speculative-num-steps 3 --speculative-draft-model-path E:/ai-models/ft-models/jzinno_Ornith-1.5-35B-A3B-DFlash2` +vision (decode,p35k) | FAIL: available RAM 0.81 GiB under 1.1 -> The guard stopped it: host RAM ran critically low. Try: close big apps, lower the context or KV, or a smaller quant.; minRAM 0.81 GiB, maxVRAM 14.47 GiB |
| 8-kv8 | `--moe-strategy hybrid --memory-ratio 0.75 --speculative-algorithm dflash --speculative-num-steps 3 --speculative-draft-model-path E:/ai-models/ft-models/jzinno_Ornith-1.5-35B-A3B-DFlash2 --kv-cache-dtype fp8_per_token_head` +vision (decode) | FAIL: available RAM 0.93 GiB under 1.1 -> The guard stopped it: host RAM ran critically low. Try: close big apps, lower the context or KV, or a smaller quant.; minRAM 0.93 GiB, maxVRAM 11.2 GiB |

## Context

- Unconfirmed context on top of the winner: **30720 tokens** (engine KV 52810 tokens; model max 262144); fill test 35K prompt

## 8-bit KV (lossy, not in the launch line)

- `--kv-cache-dtype fp8_per_token_head` did not run: available RAM 0.93 GiB under 1.1

## Expert slots against context

Picked **5000 slots** (the fastest count that holds the floor of 65536 KV tokens). Loaded at context 262144.

| slots | decode tok/s | 5K prefill tok/s | KV tokens | max VRAM GiB |
|---|---|---|---|---|
| 5000 | 77.89 | 892.3 | 113576 | 15.4 |
| 6500 | 85.4 | 906.7 | 27560 | 15.43 |

## Notes

- token map (ornith35-apex-mtp-mini.bin): 76.88 vs 73.81 tok/s, kept
- DFlash ladder k=[3, 5, 7, 11, 15] (draft block size from its config.json)
- spec decoding: MTP 76.88 tok/s, DFlash 81.26 tok/s; DFlash kept
- slot sweep: 6500 slots leave 27560 KV tokens, under the floor of 65536; kept 5000
- no tune result for class qwen35moe-moe: no flags applied. Run once with --tune (BENCH_FT_FIT_TUNE=1).
- as-stamped run with vision at --moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm dflash --speculative-num-steps 3 --speculative-draft-model-path E:/ai-models/ft-models/jzinno_Ornith-1.5-35B-A3B-DFlash2 (context 262144) failed: available RAM 1.04 GiB under 1.1 -> The guard stopped it: host RAM ran critically low. Try: close big apps, lower the context or KV, or a smaller quant.
- 51K prompt on a 52810-token context failed: available RAM 0.86 GiB under 1.1 -> The guard stopped it: host RAM ran critically low. Try: close big apps, lower the context or KV, or a smaller quant.
- 42K prompt on a 44032-token context failed: available RAM 0.81 GiB under 1.1 -> The guard stopped it: host RAM ran critically low. Try: close big apps, lower the context or KV, or a smaller quant.
- 35K prompt on a 36864-token context failed: available RAM 0.81 GiB under 1.1 -> The guard stopped it: host RAM ran critically low. Try: close big apps, lower the context or KV, or a smaller quant.
- context not confirmed by a fill test; 30720 is the KV size or the last step down
- context 30720 is under the floor of 65536 tokens: this config does not meet the goal
