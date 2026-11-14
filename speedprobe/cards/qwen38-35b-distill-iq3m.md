# Model card: qwen38-35b-distill_iq3m.gguf

Fitted 2026-11-14 11:09 by `scripts/ft-fit.py` on RX 9070 16 GB / 31.9 GiB RAM, every run at context 65536 (--max-seq-len-override).

## Recommended launch

```
ft serve --model-path "E:\ai-hot\qwen38-35b-distill_iq3m.gguf" --moe-cpu-layers auto --moe-cache-size 5000 --moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\qwen38-35b-distill-iq3m.bin
```

- Decode: **127.99 tok/s** as stamped (this exact line in one run); ladder run of the same config 127.89 tok/s (MTP, not byte-identical to plain decode); plain best 104.57 tok/s
- As-stamped run (decode + 5k prompt): decode 127.99 tok/s, p5k 108.3 tok/s; minRAM 3.71 GiB, maxVRAM 15.21 GiB
- Context floor 65536 tokens: **met** (111616 tokens fit)

## Model

- qwen35moe, 41 layers, 256 experts (8 used), file 15.21 GiB (experts 13.85, dense 1.35)
- Expert tensors: {'Q4_K': 5, 'IQ3_S': 115, 'Q8_0': 3}
- KV ~20 KiB/token (bf16 estimate): 32k context ~0.625 GiB
- Vision: none
- MTP: built-in
- Size-check verdict: tight (host need 18.85 GiB text, 18.85 with vision)

## Runs

| stage | config | result |
|---|---|---|
| 4-ratio | `--moe-strategy hybrid --memory-ratio 0.80` (decode) | decode 104.57 tok/s; minRAM 3.23 GiB, maxVRAM 14.53 GiB |
| 4-ratio | `--moe-strategy hybrid --memory-ratio 0.85` (decode) | decode 104.73 tok/s; minRAM 3.62 GiB, maxVRAM 15.32 GiB |
| 5-spec | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2` (decode) | decode 122.72 tok/s; minRAM 3.76 GiB, maxVRAM 14.61 GiB |
| 5-spec | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 3` (decode) | decode 115.99 tok/s; minRAM 3.72 GiB, maxVRAM 14.61 GiB |
| 5-spec | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\qwen38-35b-distill-iq3m.bin` (decode) | decode 127.89 tok/s; minRAM 3.7 GiB, maxVRAM 14.61 GiB |
| 6-slots | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\qwen38-35b-distill-iq3m.bin` (decode,p5k) | decode 128.37 tok/s, p5k 107.54 tok/s; minRAM 3.72 GiB, maxVRAM 15.21 GiB |
| 6-slots | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\qwen38-35b-distill-iq3m.bin --moe-cache-size 6500` (decode,p5k) | decode 137.99 tok/s, p5k 129.0 tok/s; minRAM 3.73 GiB, maxVRAM 15.19 GiB |
| 7-final | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\qwen38-35b-distill-iq3m.bin` (decode,p5k) | decode 127.99 tok/s, p5k 108.3 tok/s; minRAM 3.71 GiB, maxVRAM 15.21 GiB |
| 8-ctx | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\qwen38-35b-distill-iq3m.bin` (decode,p110k) | decode 128.56 tok/s, p110k 75.6 tok/s; minRAM 3.68 GiB, maxVRAM 15.61 GiB |
| 8-kv8 | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\qwen38-35b-distill-iq3m.bin --kv-cache-dtype fp8_per_token_head` (decode) | decode 127.19 tok/s; minRAM 3.41 GiB, maxVRAM 14.6 GiB |
| 8-kv8s | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\qwen38-35b-distill-iq3m.bin --kv-cache-dtype fp8_per_token_head --moe-cache-size 5000` (decode,p5k) | decode 122.85 tok/s, p5k 98.85 tok/s; minRAM 3.25 GiB, maxVRAM 15.29 GiB |
| 8-kv8s | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\qwen38-35b-distill-iq3m.bin --kv-cache-dtype fp8_per_token_head --moe-cache-size 6250` (decode,p5k) | decode 132.54 tok/s, p5k 104.71 tok/s; minRAM 3.13 GiB, maxVRAM 15.34 GiB |
| 8-kv8s | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\qwen38-35b-distill-iq3m.bin --kv-cache-dtype fp8_per_token_head --moe-cache-size 5000` (decode,p5k) | decode 118.66 tok/s, p5k 95.58 tok/s; minRAM 2.37 GiB, maxVRAM 15.49 GiB |
| 8-kv8s | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\qwen38-35b-distill-iq3m.bin --kv-cache-dtype fp8_per_token_head --moe-cache-size 6250` (decode,p5k) | decode 127.65 tok/s, p5k 104.4 tok/s; minRAM 2.32 GiB, maxVRAM 15.68 GiB |
| 8-kv8s | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\qwen38-35b-distill-iq3m.bin --kv-cache-dtype fp8_per_token_head --moe-cache-size 5000` (decode,p5k) | decode 118.54 tok/s, p5k 95.55 tok/s; minRAM 2.66 GiB, maxVRAM 15.69 GiB |
| 8-kv8s | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\qwen38-35b-distill-iq3m.bin --kv-cache-dtype fp8_per_token_head --moe-cache-size 6250` (decode,p5k) | decode 130.93 tok/s, p5k 99.07 tok/s; minRAM 3.01 GiB, maxVRAM 15.65 GiB |

## Context

- Confirmed context on top of the winner: **111616 tokens** (engine KV 111680 tokens; model max 262144); fill test 110K prompt, 75.6 tok/s decode at that depth

## 8-bit KV (lossy, not in the launch line)

- `--kv-cache-dtype fp8_per_token_head` on the as-stamped config: KV **219924 tokens** (16-bit: 111680), decode 127.19 tok/s (16-bit: 127.99), text DIFFERS on the decode test.
- It changes what the model sees of its own context, so adding it is your call; the decode test is too short to judge quality.

## Expert slots against context

Picked **5000 slots** (the fastest count that holds the floor of 65536 KV tokens). Loaded at context 262144.

| slots | decode tok/s | 5K prefill tok/s | KV tokens | max VRAM GiB |
|---|---|---|---|---|
| 5000 | 128.37 | 2861.2 | 111680 | 15.21 |
| 6500 | 137.99 | 2841.2 | 20451 | 15.19 |

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

- token map (qwen38-35b-distill-iq3m.bin): 127.89 vs 122.72 tok/s, kept
- slot sweep: 6500 slots leave 20451 KV tokens, under the floor of 65536; kept 5000
- flags from the tune run of 2026-11-14 06:44 on qwen38-35b-distill_iq3m.gguf (class qwen35moe-moe): none kept
