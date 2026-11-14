# Model card: Qwen3-Coder-REAP-25B-A3B-MXFP4_MOE-GGUF-Qwen3-Coder-REAP-25B-A3B-i1-codemedium-exp-MXFP4_MOE.gguf

Fitted 2026-11-13 13:35 by `scripts/ft-fit.py` on RX 9070 16 GB / 31.9 GiB RAM, every run at context 65536 (--max-seq-len-override).

## Recommended launch

```
ft serve --model-path "E:\ai-models\ft-models\noctrex_Qwen3-Coder-REAP-25B-A3B-MXFP4_MOE-GGUF\Qwen3-Coder-REAP-25B-A3B-MXFP4_MOE-GGUF-Qwen3-Coder-REAP-25B-A3B-i1-codemedium-exp-MXFP4_MOE.gguf" --moe-cpu-layers auto --moe-strategy hybrid --memory-ratio 0.80 --moe-cache-auto --kv-reserve-tokens 65536
```

- Decode: **101.45 tok/s** as stamped (this exact line in one run); ladder run of the same config 101.83 tok/s (plain decode); plain best 100.96 tok/s
- As-stamped run (decode + 5k prompt): decode 101.45 tok/s, p5k 85.65 tok/s; minRAM 3.16 GiB, maxVRAM 15.32 GiB
- Context floor 65536 tokens: **met** (65536 tokens fit)

## Model

- qwen3moe, 48 layers, 103 experts (8 used), file 13.09 GiB (experts 11.54, dense 1.55)
- Expert tensors: {'MXFP4': 144}
- KV ~96 KiB/token (bf16 estimate): 32k context ~3.0 GiB
- Vision: none
- MTP: none on disk
- Size-check verdict: hybrid (host need 16.54 GiB text, 16.54 with vision)

## Runs

| stage | config | result |
|---|---|---|
| 4-ratio | `--moe-strategy hybrid --memory-ratio 0.80` (decode) | decode 100.96 tok/s; minRAM 3.24 GiB, maxVRAM 15.22 GiB |
| 4-ratio | `--moe-strategy hybrid --memory-ratio 0.85` (decode) | decode 69.0 tok/s; minRAM 2.62 GiB, maxVRAM 15.42 GiB |
| 6-slots | `--moe-cpu-layers auto --moe-strategy hybrid --memory-ratio 0.80 --moe-cache-auto --kv-reserve-tokens 65536` (decode,p5k) | decode 101.83 tok/s, p5k 85.48 tok/s; minRAM 3.06 GiB, maxVRAM 15.32 GiB |
| 7-final | `--moe-strategy hybrid --memory-ratio 0.80 --moe-cache-auto --kv-reserve-tokens 65536` (decode,p5k) | decode 101.45 tok/s, p5k 85.65 tok/s; minRAM 3.16 GiB, maxVRAM 15.32 GiB |
| 8-ctx | `--moe-strategy hybrid --memory-ratio 0.80 --moe-cache-auto --kv-reserve-tokens 65536` (decode,p64k) | decode 99.37 tok/s, p64k 33.25 tok/s; minRAM 1.65 GiB, maxVRAM 15.33 GiB |
| 8-kv8 | `--moe-strategy hybrid --memory-ratio 0.80 --moe-cache-auto --kv-reserve-tokens 65536 --kv-cache-dtype fp8_per_token_head` (decode) | decode 100.94 tok/s; minRAM 3.81 GiB, maxVRAM 15.3 GiB |

## Context

- Confirmed context on top of the winner: **65536 tokens** (engine KV 65550 tokens; model max 262144); fill test 64K prompt, 33.25 tok/s decode at that depth

## 8-bit KV (lossy, not in the launch line)

- `--kv-cache-dtype fp8_per_token_head` on the as-stamped config: KV **65556 tokens** (16-bit: 65550), decode 100.94 tok/s (16-bit: 101.45), text DIFFERS on the decode test.
- It changes what the model sees of its own context, so adding it is your call; the decode test is too short to judge quality.

## Expert slots against context

--slots-mode auto: the engine sized the slots itself (--moe-cache-auto, KV floor 65536 tokens): **2114 slots**. Loaded at context 262144.

| slots | decode tok/s | 5K prefill tok/s | KV tokens | max VRAM GiB |
|---|---|---|---|---|
| 2114 | 101.83 | 1191.7 | 65550 | 15.32 |

## Notes

- no MTP draft head on disk; spec decoding skipped
- no tune result for class qwen3moe-moe: no flags applied. Run once with --tune (BENCH_FT_FIT_TUNE=1).
