# Model card: ornith35_apex-mtp-mini.gguf

Fitted 2026-11-14 10:09 by `scripts/ft-fit.py` on RX 9070 16 GB / 31.9 GiB RAM, every run at context 65536 (--max-seq-len-override).

## Recommended launch

```
ft serve --model-path "E:\ai-hot\ornith35_apex-mtp-mini.gguf" --moe-cpu-layers auto --moe-cache-size 5000 --moe-strategy hybrid --memory-ratio 0.80
```

- Decode: **115.03 tok/s** as stamped (this exact line in one run); ladder run of the same config 115.69 tok/s (plain decode); plain best 115.69 tok/s
- As-stamped run (decode + 5k prompt): decode 115.03 tok/s, p5k 109.26 tok/s; minRAM 3.91 GiB, maxVRAM 15.15 GiB
- Context floor 65536 tokens: **met** (222208 tokens fit)

## Model

- qwen35moe, 41 layers, 256 experts (8 used), file 13.37 GiB (experts 12.05, dense 1.32)
- Expert tensors: {'Q3_K': 60, 'IQ2_S': 60, 'Q8_0': 3}
- KV ~20 KiB/token (bf16 estimate): 32k context ~0.625 GiB
- Vision: none
- MTP: built-in
- Size-check verdict: hybrid (host need 17.05 GiB text, 17.05 with vision)

## Runs

| stage | config | result |
|---|---|---|
| 4-ratio | `--moe-strategy hybrid --memory-ratio 0.80` (decode) | decode 115.69 tok/s; minRAM 3.53 GiB, maxVRAM 14.54 GiB |
| 4-ratio | `--moe-strategy hybrid --memory-ratio 0.85` (decode) | decode 115.27 tok/s; minRAM 3.55 GiB, maxVRAM 15.33 GiB |
| 5-spec | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2` (decode) | decode 95.07 tok/s; minRAM 2.85 GiB, maxVRAM 14.62 GiB |
| 5-spec | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 3` (decode) | decode 78.71 tok/s; minRAM 2.88 GiB, maxVRAM 14.62 GiB |
| 5-spec | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\ornith35-apex-mtp-mini.bin` (decode) | decode 99.41 tok/s; minRAM 3.04 GiB, maxVRAM 14.62 GiB |
| 5-spec | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm dflash --speculative-num-steps 3 --speculative-draft-model-path E:/ai-models/ft-models/jzinno_Ornith-1.5-35B-A3B-DFlash2` (decode) | decode 98.35 tok/s; minRAM 3.6 GiB, maxVRAM 14.6 GiB |
| 5-spec | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm dflash --speculative-num-steps 5 --speculative-draft-model-path E:/ai-models/ft-models/jzinno_Ornith-1.5-35B-A3B-DFlash2` (decode) | decode 74.79 tok/s; minRAM 3.6 GiB, maxVRAM 14.6 GiB |
| 5-spec | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm dflash --speculative-num-steps 7 --speculative-draft-model-path E:/ai-models/ft-models/jzinno_Ornith-1.5-35B-A3B-DFlash2` (decode) | decode 66.09 tok/s; minRAM 3.61 GiB, maxVRAM 14.6 GiB |
| 5-spec | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm dflash --speculative-num-steps 11 --speculative-draft-model-path E:/ai-models/ft-models/jzinno_Ornith-1.5-35B-A3B-DFlash2` (decode) | decode 68.77 tok/s; minRAM 3.6 GiB, maxVRAM 14.6 GiB |
| 5-spec | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm dflash --speculative-num-steps 15 --speculative-draft-model-path E:/ai-models/ft-models/jzinno_Ornith-1.5-35B-A3B-DFlash2` (decode) | decode 62.25 tok/s; minRAM 2.69 GiB, maxVRAM 14.64 GiB |
| 6-slots | `--moe-strategy hybrid --memory-ratio 0.80` (decode,p5k) | decode 115.04 tok/s, p5k 109.15 tok/s; minRAM 3.89 GiB, maxVRAM 15.15 GiB |
| 6-slots | `--moe-strategy hybrid --memory-ratio 0.80 --moe-cache-size 6500` (decode,p5k) | decode 113.05 tok/s, p5k 109.61 tok/s; minRAM 3.25 GiB, maxVRAM 15.15 GiB |
| 7-final | `--moe-strategy hybrid --memory-ratio 0.80` (decode,p5k) | decode 115.03 tok/s, p5k 109.26 tok/s; minRAM 3.91 GiB, maxVRAM 15.15 GiB |
| 8-ctx | `--moe-strategy hybrid --memory-ratio 0.80` (decode,p221k) | decode 115.13 tok/s, p221k 57.87 tok/s; minRAM 3.81 GiB, maxVRAM 15.53 GiB |
| 8-kv8 | `--moe-strategy hybrid --memory-ratio 0.80 --kv-cache-dtype fp8_per_token_head` (decode) | decode 108.97 tok/s; minRAM 3.95 GiB, maxVRAM 14.56 GiB |

## Context

- Confirmed context on top of the winner: **222208 tokens** (engine KV 222691 tokens; model max 262144); fill test 221K prompt, 57.87 tok/s decode at that depth

## 8-bit KV (lossy, not in the launch line)

- `--kv-cache-dtype fp8_per_token_head` on the as-stamped config: KV **438530 tokens** (16-bit: 222691), decode 108.97 tok/s (16-bit: 115.03), text DIFFERS on the decode test.
- It changes what the model sees of its own context, so adding it is your call; the decode test is too short to judge quality.

## Expert slots against context

Picked **5000 slots** (the fastest count that holds the floor of 65536 KV tokens). Loaded at context 262144.

| slots | decode tok/s | 5K prefill tok/s | KV tokens | max VRAM GiB |
|---|---|---|---|---|
| 5000 | 115.04 | 2705.6 | 222691 | 15.15 |
| 6500 | 113.05 | 2919.0 | 136675 | 15.15 |

Flags below come from the tune run of 2026-11-14 06:44 on qwen38-35b-distill_iq3m.gguf (class qwen35moe-moe); this fit only ran them in the as-stamped run.

## Parallel requests

One request decodes at 96.59 tok/s. Best: **8 at once**. Not in the launch line: the context per request is your call. To use N: `--max-running-requests N --cuda-graph-max-bs N` and FT_GRAPH_MAX_BS=N.

| requests | graph bs | extra args | total tok/s | per request tok/s | one request tok/s | KV tokens | max VRAM GiB | min RAM GiB |
|---|---|---|---|---|---|---|---|---|
| 2 | 2 |  | 168.37 | 84.02 84.25 | 103.66 | 214339 | 14.5 | 1.17 |
| 4 | 4 |  | 223.19 | 55.69 55.8 55.8 55.8 | 104.13 | 176611 | 14.55 | 1.45 |
| 8 | 8 |  | 264.71 | 33.03 33.07 33.07 33.07 33.07 33.07 33.07 33.07 | 103.88 | 101155 | 14.58 | 1.72 |
| 4 | 1 |  | 56.87 | 14.19 14.26 14.26 14.26 | 103.81 | 176611 | 14.51 | 1.81 |

## Notes

- token map (ornith35-apex-mtp-mini.bin): 99.41 vs 95.07 tok/s, kept
- DFlash ladder k=[3, 5, 7, 11, 15] (draft block size from its config.json)
- spec decoding: MTP 99.41 tok/s, DFlash 98.35 tok/s; MTP kept
- slot sweep: 6500 slots is no faster (113.05 vs 115.04 tok/s); kept 5000
- flags from the tune run of 2026-11-14 06:44 on qwen38-35b-distill_iq3m.gguf (class qwen35moe-moe): none kept
