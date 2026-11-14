# Model card: Qwopus3.6-35B-A3B-v1-MTP-Q3_K_M.gguf

Fitted 2026-11-13 05:53 by `scripts/ft-fit.py` on RX 9070 16 GB / 31.9 GiB RAM, every run at context 65536 (--max-seq-len-override).

## Recommended launch

```
ft serve --model-path "E:\ai-models\ft-models\Jackrong_Qwopus3.6-35B-A3B-v1-MTP-GGUF\Qwopus3.6-35B-A3B-v1-MTP-Q3_K_M.gguf" --moe-cpu-layers auto --moe-cache-size 5000 --moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\qwopus3-6-35b-a3b-v1-mtp-q3-k-m.bin --mm-disable vision
```

- Decode: **133.25 tok/s** as stamped (this exact line in one run); ladder run of the same config 137.25 tok/s (MTP, not byte-identical to plain decode); plain best 90.91 tok/s
- As-stamped run (decode + 5k prompt): decode 133.25 tok/s, p5k 99.94 tok/s; minRAM 2.11 GiB, maxVRAM 15.53 GiB
- Context floor 65536 tokens: **met** (100352 tokens fit)

## Model

- qwen35moe, 41 layers, 256 experts (8 used), file 15.98 GiB (experts 14.64, dense 1.34)
- Expert tensors: {'Q5_K': 2, 'Q3_K': 82, 'Q4_K': 39}
- KV ~20 KiB/token (bf16 estimate): 32k context ~0.625 GiB
- Vision: mmproj-F32.gguf (0.84 GiB)
- MTP: built-in
- Size-check verdict: tight (host need 19.64 GiB text, 20.48 with vision)

## Runs

| stage | config | result |
|---|---|---|
| 4-ratio | `--moe-strategy hybrid --memory-ratio 0.80` (decode) | decode 90.91 tok/s; minRAM 1.4 GiB, maxVRAM 15.43 GiB |
| 4-ratio | `--moe-strategy hybrid --memory-ratio 0.85` (decode) | decode 45.4 tok/s; minRAM 1.9 GiB, maxVRAM 15.56 GiB |
| 5-spec | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2` (decode) | decode 125.75 tok/s; minRAM 2.46 GiB, maxVRAM 15.42 GiB |
| 5-spec | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 3` (decode) | decode 118.92 tok/s; minRAM 1.92 GiB, maxVRAM 15.41 GiB |
| 5-spec | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\qwopus3-6-35b-a3b-v1-mtp-q3-k-m.bin` (decode) | decode 137.25 tok/s; minRAM 2.04 GiB, maxVRAM 15.42 GiB |
| 6-slots | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\qwopus3-6-35b-a3b-v1-mtp-q3-k-m.bin` (decode,p5k) | decode 138.11 tok/s, p5k 119.52 tok/s; minRAM 2.01 GiB, maxVRAM 15.54 GiB |
| 6-slots | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\qwopus3-6-35b-a3b-v1-mtp-q3-k-m.bin --moe-cache-size 6500` (decode,p5k) | decode 140.17 tok/s, p5k None tok/s; minRAM 1.81 GiB, maxVRAM 15.4 GiB |
| 7-final | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\qwopus3-6-35b-a3b-v1-mtp-q3-k-m.bin` +vision (decode,p5k) | FAIL: available RAM 0.97 GiB under 1.1 -> The guard stopped it: host RAM ran critically low. Try: close big apps, lower the context or KV, or a smaller quant.; minRAM 0.97 GiB, maxVRAM 11.84 GiB |
| 7-final | `--moe-strategy hybrid --memory-ratio 0.75 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\qwopus3-6-35b-a3b-v1-mtp-q3-k-m.bin` +vision (decode,p5k) | FAIL: available RAM 0.89 GiB under 1.1 -> The guard stopped it: host RAM ran critically low. Try: close big apps, lower the context or KV, or a smaller quant.; minRAM 0.89 GiB, maxVRAM 14.31 GiB |
| 7-final | `--moe-strategy hybrid --memory-ratio 0.70 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\qwopus3-6-35b-a3b-v1-mtp-q3-k-m.bin` +vision (decode,p5k) | FAIL: available RAM 0.90 GiB under 1.1 -> The guard stopped it: host RAM ran critically low. Try: close big apps, lower the context or KV, or a smaller quant.; minRAM 0.9 GiB, maxVRAM 13.5 GiB |
| 7-final | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\qwopus3-6-35b-a3b-v1-mtp-q3-k-m.bin` (decode,p5k) | decode 133.25 tok/s, p5k 99.94 tok/s; minRAM 2.11 GiB, maxVRAM 15.53 GiB |
| 8-ctx | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\qwopus3-6-35b-a3b-v1-mtp-q3-k-m.bin` (decode,p99k) | decode 134.37 tok/s, p99k 77.96 tok/s; minRAM 2.04 GiB, maxVRAM 15.56 GiB |
| 8-kv8 | `--moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\qwopus3-6-35b-a3b-v1-mtp-q3-k-m.bin --kv-cache-dtype fp8_per_token_head` (decode) | decode 127.75 tok/s; minRAM 2.64 GiB, maxVRAM 15.4 GiB |

## Context

- Confirmed context on top of the winner: **100352 tokens** (engine KV 101254 tokens; model max 262144); fill test 99K prompt, 77.96 tok/s decode at that depth

## 8-bit KV (lossy, not in the launch line)

- `--kv-cache-dtype fp8_per_token_head` on the as-stamped config: KV **199392 tokens** (16-bit: 101254), decode 127.75 tok/s (16-bit: 133.25), text DIFFERS on the decode test.
- It changes what the model sees of its own context, so adding it is your call; the decode test is too short to judge quality.

## Expert slots against context

Picked **5000 slots** (the fastest count that holds the floor of 65536 KV tokens). Loaded at context 262144.

| slots | decode tok/s | 5K prefill tok/s | KV tokens | max VRAM GiB |
|---|---|---|---|---|
| 5000 | 138.11 | 1248.3 | 101254 | 15.54 |
| 6500 | 140.17 |  | 1646 | 15.4 |

## Notes

- token map (qwopus3-6-35b-a3b-v1-mtp-q3-k-m.bin): 137.25 vs 125.75 tok/s, kept
- slot sweep: 6500 slots leave 1646 KV tokens, under the floor of 65536; kept 5000
- no tune result for class qwen35moe-moe: no flags applied. Run once with --tune (BENCH_FT_FIT_TUNE=1).
- as-stamped run with vision at --moe-strategy hybrid --memory-ratio 0.80 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\qwopus3-6-35b-a3b-v1-mtp-q3-k-m.bin (context 262144) failed: available RAM 0.97 GiB under 1.1 -> The guard stopped it: host RAM ran critically low. Try: close big apps, lower the context or KV, or a smaller quant.
- as-stamped run with vision at --moe-strategy hybrid --memory-ratio 0.75 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\qwopus3-6-35b-a3b-v1-mtp-q3-k-m.bin (context 262144) failed: available RAM 0.89 GiB under 1.1 -> The guard stopped it: host RAM ran critically low. Try: close big apps, lower the context or KV, or a smaller quant.
- as-stamped run with vision at --moe-strategy hybrid --memory-ratio 0.70 --speculative-algorithm mtp --speculative-num-steps 2 --speculative-token-map E:\freetoken-021\speedprobe\token-maps\qwopus3-6-35b-a3b-v1-mtp-q3-k-m.bin (context 262144) failed: available RAM 0.90 GiB under 1.1 -> The guard stopped it: host RAM ran critically low. Try: close big apps, lower the context or KV, or a smaller quant.
- vision does not fit with this config: launch with --mm-disable vision
