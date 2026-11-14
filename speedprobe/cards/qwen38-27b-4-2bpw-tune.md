# Tune result: class qwen35-dense (run on qwen38-27b_4.2bpw.gguf)

Fitted 2026-11-14 06:30 by `scripts/ft-fit.py` on RX 9070 16 GB / 31.9 GiB RAM, every run at context 65536 (--max-seq-len-override).

**Not a config to stamp.** Flags and the parallel report for every model of this class, measured on the plain config `--memory-ratio 0.90 --max-running-requests 1` (33.66 tok/s). A fit of such a model applies the kept flags in its as-stamped run.

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
| 4-dense | `--memory-ratio 0.90` (decode) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-qwen38-27b-4-2bpw-dense-r90.log; minRAM 2.83 GiB, maxVRAM 13.27 GiB |
| 4-dense | `--memory-ratio 0.90 --max-running-requests 1` (decode) | decode 33.66 tok/s; minRAM 2.25 GiB, maxVRAM 15.68 GiB |
| 6-flag | `--memory-ratio 0.90 --max-running-requests 1` (decode,p5k) | decode 33.53 tok/s, p5k 32.95 tok/s; minRAM 1.53 GiB, maxVRAM 15.69 GiB |
| 6-flag | `--memory-ratio 0.90 --max-running-requests 1 --max-prefill-length 2048` (decode,p5k) | decode 33.65 tok/s, p5k 33.05 tok/s; minRAM 2.07 GiB, maxVRAM 15.69 GiB |
| 6-flag | `--memory-ratio 0.90 --max-running-requests 1 --max-prefill-length 4096` (decode,p5k) | decode 33.67 tok/s, p5k 33.06 tok/s; minRAM 1.72 GiB, maxVRAM 15.69 GiB |
| 6-flag | `--memory-ratio 0.90 --max-running-requests 1 --max-prefill-length 2048` (decode,p5k) | decode 33.76 tok/s, p5k 33.28 tok/s; minRAM 2.18 GiB, maxVRAM 15.69 GiB |
| 6p-parallel | `--memory-ratio 0.90 --max-running-requests 2 --cuda-graph-max-bs 2` {'FT_GRAPH_MAX_BS': '2'} (decode,c2) | decode 33.79 tok/s, c2 60.11 tok/s; minRAM 2.37 GiB, maxVRAM 15.68 GiB |
| 6p-parallel | `--memory-ratio 0.90 --max-running-requests 4 --cuda-graph-max-bs 4` {'FT_GRAPH_MAX_BS': '4'} (decode,c4) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-qwen38-27b-4-2bpw-dense-r90-mr1-par4.log; minRAM 3.54 GiB, maxVRAM 12.37 GiB |
| 6p-parallel | `--memory-ratio 0.90 --max-running-requests 4 --cuda-graph-max-bs 1` {'FT_GRAPH_MAX_BS': '1'} (decode,c4) | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-qwen38-27b-4-2bpw-dense-r90-mr1-par4-g1.log; minRAM 3.86 GiB, maxVRAM 12.52 GiB |

## Flags

- `cpu-threads-6` (safe, lossless True): not run (MoE-only flag; dense model)
- `cpu-threads-8` (safe, lossless True): not run (MoE-only flag; dense model)
- `fused-combine` (candidate, lossless unknown): not run (candidate)
- `hybrid-max-fetch` (safe, lossless True): not run (MoE-only flag; dense model)
- `locked-max-fetch` (safe, lossless True): not run (MoE-only flag; dense model)
- `prefill-chunk` (safe, lossless True): `--max-prefill-length 2048`: decode 33.65 tok/s, p5k 33.05 tok/s; minRAM 2.07 GiB, maxVRAM 15.69 GiB; prefill:p5k 362.4 vs 193.5; text identical | `--max-prefill-length 4096`: decode 33.67 tok/s, p5k 33.06 tok/s; minRAM 1.72 GiB, maxVRAM 15.69 GiB; prefill:p5k 229.6 vs 193.5; text identical | second run | `--max-prefill-length 2048`: decode 33.76 tok/s, p5k 33.28 tok/s; minRAM 2.18 GiB, maxVRAM 15.69 GiB; prefill:p5k 357.2 vs 193.5; text identical **kept**
- `cache-scope-per-layer` (safe, lossless True): not run (MoE-only flag; dense model)
- `prefill-hit-d2d` (candidate, lossless True): not run (MoE-only flag; dense model)
- `prefill-no-overlap` (candidate, lossless True): not run (MoE-only flag; dense model)
- `cache-split-profile` (candidate, lossless True): not run (MoE-only flag; dense model)
- `kv-cache-fp8` (candidate, lossless False): not run (candidate)
- `spec-token-map` (candidate, lossless True): not run (candidate)
- `online-quant-df11` (candidate, lossless True): not run (candidate)

## Parallel requests

One request decodes at 33.66 tok/s. Best: **2 at once**. Not in the launch line: the context per request is your call. To use N: `--max-running-requests N --cuda-graph-max-bs N` and FT_GRAPH_MAX_BS=N.

| requests | graph bs | extra args | total tok/s | per request tok/s | one request tok/s | KV tokens | max VRAM GiB | min RAM GiB |
|---|---|---|---|---|---|---|---|---|
| 2 | 2 |  | 60.11 | 30.01 30.0 | 33.79 | 9053 | 15.68 | 2.37 |
| 4 | 4 |  | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-qwen38-27b-4 |  |  |  | 12.37 | 3.54 |
| 4 | 1 |  | FAIL: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-qwen38-27b-4 |  |  |  | 12.52 | 3.86 |

## Notes

- dense --memory-ratio 0.90 failed: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-qwen38-27b-4-2bpw-dense-r90.log
- parallel stage: dense-r90-mr1-par4 failed: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-qwen38-27b-4-2bpw-dense-r90-mr1-par4.log
- parallel stage: dense-r90-mr1-par4-g1 failed: server exited (15) during load; see E:\freetoken-021\speedprobe\fit-qwen38-27b-4-2bpw-dense-r90-mr1-par4-g1.log
- tune run: flags and parallel report saved to E:\freetoken-021\speedprobe\tune\qwen35-dense.json; not a config to stamp
