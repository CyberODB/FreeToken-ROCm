<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo-light.svg">
    <img alt="FreeToken" src="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo.svg" width=65%>
  </picture>
</div>

> [!NOTE]
> **This branch (`win-rocm-021`): Windows + AMD ROCm test build on [FreeToken](https://github.com/FlashML-org/FreeToken) v0.2.1.**
> FreeToken is by FlashML-org (Apache 2.0), and v0.2.1 includes upstream's own ROCm support.
> Maxritz's [FreeToken-ROCm](https://github.com/Maxritz/FreeToken-ROCm) first showed that FreeToken could run on Windows with AMD ROCm,
> and this branch would not exist without it. The Windows/ROCm work here started from his fork
> (with contributions from PialGhosh2233, Castoff995, Elliot, Aron van Ammers, Roman Baumer and Daniel Bittencourt);
> parts of it are re-applied here on top of v0.2.1, with Windows daemon, launcher and RX 9070 fixes
> by CyberODB, made with help from Claude. The older port with Maxritz's full history is on `main`.

## Windows + ROCm milestones (this branch)

Measured on an AMD Radeon RX 9070 (gfx1201, 16 GB), Windows 11, 32 GB RAM, model on a SATA SSD,
with HauhauCS Qwen3.6-35B-A3B IQ4_NL GGUF in hybrid MoE mode (experts in host RAM, run on the CPU and GPU).

| What | Before | After |
|---|---|---|
| Hybrid decode (HIP graphs, batch 1) | 12-14.5 tok/s, graphs forced off | **47-58 tok/s**, output coherent |
| Expert banks load (16.9 GiB) | 86-87 s at ~210 MiB/s | **35 s** at 443-447 MiB/s |
| GGUF header parse (model file) | 10.7-11.4 s, run 3 times per launch | **1.2 s**, parsed once per process |
| Server argument parsing | 10.6 s | **1.6-1.9 s** |
| `transformers` package scan at import, per process | 2.85 s | **0.16 s** |
| Dense + vision weights, cold disk (2.37 GiB) | 11.1 s | **7.4 s** |

- **HIP graphs in hybrid decode.** On Windows ROCm, stream memory-op graph nodes never run on replay, so the
  CPU-MoE handshake fell back to host callbacks and graphs were turned off. A one-thread GPU kernel now spins
  on the CPU's done flag instead, which is graph-safe. Graphs are on by default at batch 1. Turn them off
  with `FT_GRAPH_MAX_BS=0`, or `FreeToken-Desktop-021.cmd 0`. `FREETOKEN_CPU_MOE_KERNEL_WAIT=0` opts out of
  the kernel wait.
- **Threaded expert bank reads.** Windows has no `O_DIRECT` or `preadv`, so the banks were filled from the
  file mmap one page fault at a time. 8 threads now read each expert's rows straight into the pinned bank.
  `FREETOKEN_GGUF_READ_THREADS=0` restores the mmap copy.
- **Faster startup.** The vendored gguf reader indexes a plain ndarray view of the file map and reads the
  tokenizer's string arrays without recursing per item. `freetoken._fast_metadata` gives transformers a
  `packages_distributions()` that skips ~26k file `stat` calls. 8 background threads read the dense
  weights ahead of the loader; `FREETOKEN_GGUF_PREFETCH_THREADS=0` turns that off. Parsed metadata,
  package mappings and loaded bytes were checked identical to the old code paths.
- The threaded reads, the prefetch and the metadata patch are off outside Windows. The kernel wait only
  runs on HIP when the memory-op probe fails.

<p align="center">
| <a href="https://www.flashml.ai/"><b>Download</b></a> | <a href="https://arxiv.org/abs/2608.16157"><b>Paper</b></a> | <a href="https://join.slack.com/t/flashml/shared_invite/zt-3zpdh5j10-9dwTXrgLiqpVxizhA9KVbA"><b>Developer Slack</b></a> | <a href="https://discord.gg/MsA277cJzZ"><b>Community Discord</b></a> | <a href="https://github.com/FlashML-org/FreeToken/issues/482"><b>Community WeChat</b></a> |
</p>


Unlock datacenter-class intelligence on the hardware you already own — Run 290B+ frontier MoE models locally on your gaming PC at blistering interactive speeds.

## About

FreeToken is an edge-native Mixture-of-Experts (MoE) serving engine designed for running frontier-scale open-weight models on personal and consumer hardware. It treats heterogeneous edge resources—GPUs, CPUs, host memory, and interconnects—as a unified, elastic inference platform. Its core features include:  

- **Fast Edge-Native Runtime**: Provides efficient MoE serving with bandwidth-adaptive CPU–GPU co-execution ($q^\star$ policy), full-layer double-buffered prefill streaming, global LRU expert caching, graph-compatible execution, and the FTW fast weight format.  
- **Semantic-Aware Caching**: Features semantic anchor checkpoints for recurrent state and KV caches, allowing agentic context edits (e.g., tool calls, thinking blocks) to avoid redundant context recomputation.  
- **Elastic Memory Management**: Supports dynamic, runtime VRAM re-allocation between expert caches and KV memory without engine restarts or weight reloading.  
- **Broad MoE & Ecosystem Support**: Supports frontier open-weight MoE models (e.g., DeepSeek-V4-Flash, Qwen3.6-35B-A3B, GLM-5.2) across various parameter scales and quantization formats (e.g., MXFP4, NVFP4, FP8, BF16, GGUF), with Anthropic/OpenAI-compatible APIs for seamless integration with real-world coding and tool-calling agents (e.g., Codex, Claude Code, OpenCode, OpenClaw, DeepSeek Harness). 
- **Diverse Consumer Hardware**: Scales across consumer laptops, gaming desktops, and workstation GPUs, with native support for NVIDIA RTX 30, RTX 40, and RTX 50 series GPUs.  

## Getting Started

### Desktop app

Download FreeToken for Windows or Linux at [flashml.ai](https://www.flashml.ai/). It sets the engine up for you and gives you a GUI for running models, chatting, and tuning the engine.

<div align="center">
  <img alt="FreeToken Desktop" src="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/desktop-console.png" width=92%>
</div>

### CLI

Install FreeToken with [uv](https://docs.astral.sh/uv/) (recommended) or pip:

```bash
uv pip install "freetoken[accel]"
```

Or build from source:

```bash
git clone https://github.com/FlashML-org/FreeToken.git && cd FreeToken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

For More details:

- [Install FreeToken](https://github.com/FlashML-org/FreeToken/blob/main/docs/install.md)
- [Quick start](https://github.com/FlashML-org/FreeToken/blob/main/docs/quickstart.md)
- [Supported models](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md)
- [CLI reference](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md)
- [Repairing old FTW checkpoints](https://github.com/FlashML-org/FreeToken/blob/main/docs/ftw-hotfix.md)
- [Diagnosing a checkpoint that fails to load](https://github.com/FlashML-org/FreeToken/blob/main/scripts/checkpoint_diagnose/README.md)

## Citation

If you use FreeToken for your research, please cite our [paper](https://arxiv.org/abs/2608.16157):

```bibtex
@article{yang2026freetoken,
  title={FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution},
  author={Yang, Shuo and Fan, Xiaoze and Pan, Melissa and Xi, Haocheng and Wang, Zhe and Sun, Shanlin and Keutzer, Kurt and Han, Song and Zaharia, Matei and Xu, Chenfeng and Stoica, Ion},
  journal={arXiv preprint arXiv:2608.16157},
  year={2026}
}
```

## Acknowledgment

FreeToken was deeply inspired by [mini-sglang](https://github.com/sgl-project/mini-sglang), and
learned the design and reused code from the following projects:
[SGLang](https://github.com/sgl-project/sglang),
[vLLM](https://github.com/vllm-project/vllm),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention),
[LightLLM](https://github.com/ModelTC/lightllm) and [llama.cpp](https://github.com/ggml-org/llama.cpp).

## License

[Apache License 2.0](https://github.com/FlashML-org/FreeToken/blob/main/LICENSE).
